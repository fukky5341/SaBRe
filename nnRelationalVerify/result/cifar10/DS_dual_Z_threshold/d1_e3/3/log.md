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
execution time: IAR + RelationalAnalysis = 7.75 + 190.50 = 198.25 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0213943, upper bound: 0.0213963

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3591

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 426

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213909, upper bound: 0.0210651
time: 7.25 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0210606, upper bound: 0.0213941
time: 9.63 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 16.96 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 16.96
Output dim: 3, lower bound: -0.0213909, upper bound: 0.0210651
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 16.96
Output dim: 3, lower bound: -0.0210606, upper bound: 0.0213941

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3686839, 0.3686810
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4844899, 0.4844472
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1716030, 0.1716005
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1023657, 0.1023313
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0737663, 0.0737735
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1340476, 0.1340272
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0267763, 0.0267789
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1099969, 0.1100081
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.4109640, 0.4109477
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4642785, 0.4642248

Time for backsubstitution: 5.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3591

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 497

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213914, upper bound: 0.0210650
time: 6.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213913, upper bound: 0.0210603
time: 52.24 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3686810, 0.3686839
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4844472, 0.4844899
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1716005, 0.1716030
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1023313, 0.1023657
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0737735, 0.0737663
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1340272, 0.1340476
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0267789, 0.0267763
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1100081, 0.1099968
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.4109477, 0.4109640
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4642247, 0.4642785

Time for backsubstitution: 5.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3591

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 497

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0210610, upper bound: 0.0210605
time: 78.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0210602, upper bound: 0.0213898
time: 157.54 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 241.78 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 241.78
Output dim: 3, lower bound: -0.0213914, upper bound: 0.0210650
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 241.78
Output dim: 3, lower bound: -0.0213913, upper bound: 0.0210603
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 241.78
Output dim: 3, lower bound: -0.0210610, upper bound: 0.0210605
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 241.78
Output dim: 3, lower bound: -0.0210602, upper bound: 0.0213898

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3687624, 0.3687580
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4844680, 0.4844248
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1716374, 0.1716349
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1023756, 0.1023413
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0737371, 0.0737436
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1340331, 0.1340129
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0267547, 0.0267565
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1099906, 0.1100019
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.4109553, 0.4109390
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4642996, 0.4642462

Time for backsubstitution: 5.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3591

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 513

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213886, upper bound: 0.0210565
time: 70.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213860, upper bound: 0.0210602
time: 254.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3687609, 0.3687595
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4844675, 0.4844253
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1716374, 0.1716349
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1023756, 0.1023412
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0737364, 0.0737444
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1340332, 0.1340127
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0267539, 0.0267573
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1099907, 0.1100019
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.4109553, 0.4109390
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4643000, 0.4642457

Time for backsubstitution: 5.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3591

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 513

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213889, upper bound: 0.0210587
time: 28.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213848, upper bound: 0.0210610
time: 7.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3687580, 0.3687624
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4844248, 0.4844680
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1716349, 0.1716374
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1023413, 0.1023756
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0737436, 0.0737371
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1340129, 0.1340331
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0267565, 0.0267547
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1100019, 0.1099906
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.4109390, 0.4109553
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4642462, 0.4642996

Time for backsubstitution: 5.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3591

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 513

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0210578, upper bound: 0.0213883
time: 9.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0210557, upper bound: 0.0213951
time: 4.73 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 19.31 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 19.31
Output dim: 3, lower bound: -0.0213886, upper bound: 0.0210565
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 19.31
Output dim: 3, lower bound: -0.0213860, upper bound: 0.0210602
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 19.31
Output dim: 3, lower bound: -0.0213889, upper bound: 0.0210587
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 19.31
Output dim: 3, lower bound: -0.0213848, upper bound: 0.0210610
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 19.31
Output dim: 3, lower bound: -0.0210578, upper bound: 0.0213883
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 19.31
Output dim: 3, lower bound: -0.0210557, upper bound: 0.0213951

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3689917, 0.3689743
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4844202, 0.4843767
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1708359, 0.1708588
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1022512, 0.1022142
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0736955, 0.0737014
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1340763, 0.1340504
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0263165, 0.0263032
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1089671, 0.1089490
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.4107304, 0.4107084
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4642408, 0.4641876

Time for backsubstitution: 5.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3591

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3352

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213869, upper bound: 0.0210443
time: 14.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213749, upper bound: 0.0210537
time: 6.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3689787, 0.3689871
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4844198, 0.4843771
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1708613, 0.1708334
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1022485, 0.1022169
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0736949, 0.0737020
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1340706, 0.1340561
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0263014, 0.0263184
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1089377, 0.1089784
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.4107246, 0.4107141
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4642411, 0.4641874

Time for backsubstitution: 5.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3591

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3352

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213823, upper bound: 0.0210490
time: 57.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213698, upper bound: 0.0210588
time: 6.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3689901, 0.3689758
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4844198, 0.4843771
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1708359, 0.1708588
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1022513, 0.1022142
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0736948, 0.0737021
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1340765, 0.1340502
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0263158, 0.0263040
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1089671, 0.1089490
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.4107304, 0.4107083
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4642412, 0.4641872

Time for backsubstitution: 5.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3591

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3352

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213855, upper bound: 0.0210470
time: 5.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213755, upper bound: 0.0210541
time: 6.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3689772, 0.3689888
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4844193, 0.4843776
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1708613, 0.1708334
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1022485, 0.1022169
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0736942, 0.0737027
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1340708, 0.1340560
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0263006, 0.0263191
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1089377, 0.1089783
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.4107246, 0.4107140
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4642414, 0.4641870

Time for backsubstitution: 5.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3591

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3352

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213821, upper bound: 0.0210517
time: 5.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213695, upper bound: 0.0210473
time: 149.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3689872, 0.3689787
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4843771, 0.4844198
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1708334, 0.1708613
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1022169, 0.1022485
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0737020, 0.0736949
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1340561, 0.1340706
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0263184, 0.0263014
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1089784, 0.1089377
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.4107141, 0.4107246
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4641874, 0.4642410

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3591

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3352

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0210542, upper bound: 0.0210423
time: 111.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0210478, upper bound: 0.0213829
time: 116.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3689743, 0.3689917
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4843767, 0.4844203
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1708588, 0.1708359
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1022142, 0.1022512
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0737014, 0.0736955
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1340504, 0.1340763
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0263032, 0.0263165
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1089490, 0.1089671
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.4107084, 0.4107304
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4641876, 0.4642408

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3591

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3352

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0210495, upper bound: 0.0213765
time: 27.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0210425, upper bound: 0.0210492
time: 213.11 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 246.50 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 246.50
Output dim: 3, lower bound: -0.0213869, upper bound: 0.0210443
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 246.50
Output dim: 3, lower bound: -0.0213749, upper bound: 0.0210537
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 246.50
Output dim: 3, lower bound: -0.0213823, upper bound: 0.0210490
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 246.50
Output dim: 3, lower bound: -0.0213698, upper bound: 0.0210588
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 246.50
Output dim: 3, lower bound: -0.0213855, upper bound: 0.0210470
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 246.50
Output dim: 3, lower bound: -0.0213755, upper bound: 0.0210541
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 246.50
Output dim: 3, lower bound: -0.0213821, upper bound: 0.0210517
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 246.50
Output dim: 3, lower bound: -0.0213695, upper bound: 0.0210473
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 246.50
Output dim: 3, lower bound: -0.0210542, upper bound: 0.0210423
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 246.50
Output dim: 3, lower bound: -0.0210478, upper bound: 0.0213829
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 246.50
Output dim: 3, lower bound: -0.0210495, upper bound: 0.0213765
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 246.50
Output dim: 3, lower bound: -0.0210425, upper bound: 0.0210492

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3689916, 0.3689742
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4844191, 0.4843752
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1708359, 0.1708587
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1022511, 0.1022141
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0736954, 0.0737014
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1340763, 0.1340504
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0263165, 0.0263032
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1089670, 0.1089490
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.4107303, 0.4107077
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4642407, 0.4641871

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3591

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 636

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213736, upper bound: 0.0210455
time: 9.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213854, upper bound: 0.0210350
time: 96.76 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3689916, 0.3689742
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4844186, 0.4843755
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1708359, 0.1708587
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1022511, 0.1022141
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0736955, 0.0737014
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1340763, 0.1340504
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0263165, 0.0263032
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1089671, 0.1089489
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.4107296, 0.4107083
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4642404, 0.4641875

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3242
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2595
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 348
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3483
type: DSZ, layer: 1, pos: 3258
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2536
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 3128
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 737
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 603
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 41
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 292
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 442
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 532
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 564
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 669
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 674
type: DSZ, layer: 1, pos: 682
type: DSZ, layer: 1, pos: 700
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2078
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2323
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2365
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2561
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2579
type: DSZ, layer: 1, pos: 2690
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2699
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2935
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3066
type: DSZ, layer: 1, pos: 3085
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3103
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3140
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3217
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3305
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3544
type: DSZ, layer: 1, pos: 3554
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3591

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 636

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213611, upper bound: 0.0210530
time: 6.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0213748, upper bound: 0.0210436
time: 6.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.5930204, -2.7533655, -3.5930204, -2.7533655, -0.3689786, 0.3689871
1: -5.0398097, -3.9154196, -5.0398097, -3.9154196, -0.4844187, 0.4843755
2: -0.1377029, 0.1245110, -0.1377029, 0.1245110, -0.1708612, 0.1708333
3: -1.2741715, -0.8998085, -1.2741715, -0.8998085, -0.1022484, 0.1022168
4: 0.0945238, 0.3353949, 0.0945238, 0.3353949, -0.0736949, 0.0737019
5: -1.4164877, -1.0170685, -1.4164877, -1.0170685, -0.1340706, 0.1340561
6: 0.4208454, 0.5605581, 0.4208454, 0.5605581, -0.0263014, 0.0263183
7: -2.1314387, -1.6105540, -2.1314387, -1.6105540, -0.1089376, 0.1089784
8: -4.8678236, -4.0497561, -4.8678236, -4.0497561, -0.4107246, 0.4107134
9: -4.6579103, -3.8220367, -4.6579103, -3.8220367, -0.4642410, 0.4641870

Time for backsubstitution: 5.48 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 198.25 + 1606.67 = 1804.93 seconds
