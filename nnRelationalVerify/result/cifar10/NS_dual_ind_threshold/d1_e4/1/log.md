## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 1)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.053238807900000004


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2.8359499, -1.0793538, -2.8359499, -1.0793538, -0.9605150, 0.9605150)
1: (-3.5325108, -1.3292351, -3.5325108, -1.3292351, -1.4482784, 1.4482784)
2: (-1.7521802, -1.0798626, -1.7521802, -1.0798626, -0.2850765, 0.2850764)
3: (-1.3847566, -0.3777040, -1.3847566, -0.3777040, -0.4556699, 0.4556699)
4: (-0.9554005, -0.2971784, -0.9554005, -0.2971784, -0.3860793, 0.3860793)
5: (-1.0003092, 0.0410480, -1.0003092, 0.0410480, -0.4184362, 0.4184362)
6: (-2.4380598, -1.4722536, -2.4380598, -1.4722536, -0.3700601, 0.3700601)
7: (0.8476936, 1.2704575, 0.8476936, 1.2704575, -0.1624704, 0.1624704)
8: (-3.9828970, -2.4737611, -3.9828970, -2.4737611, -0.7428186, 0.7428186)
9: (-4.4693017, -2.7444663, -4.4693017, -2.7444663, -1.2637931, 1.2637930)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.77 + 127.25 = 135.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0532849, upper bound: 0.0532896

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 428
type: A, layer: 1, pos: 382
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 375
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 411
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 3507
type: A, layer: 1, pos: 265
type: A, layer: 1, pos: 363
type: A, layer: 1, pos: 318
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 3568
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 376
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 3321
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 385
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 425
type: A, layer: 1, pos: 3037
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 3485
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 287
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 3433
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 3300
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3452
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 413
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 3303
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 3561
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 3576
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 3580
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 3578
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 3335
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 461
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2434
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 192
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 3559
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2335
type: A, layer: 1, pos: 3352
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 2336
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 3010
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 3574
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 493
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3575
type: A, layer: 1, pos: 3358
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 3557
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 436
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2462
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2469
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3239
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3362
type: A, layer: 1, pos: 3365
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3569
type: A, layer: 1, pos: 3585

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 428

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0530600, upper bound: 0.0532844
time: 30.46 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532856, upper bound: 0.0532912
time: 207.99 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 238.51 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 238.51
Output dim: 7, lower bound: -0.0530600, upper bound: 0.0532844
NS_A2, status: Status.UNKNOWN, split count: 1, time: 238.51
Output dim: 7, lower bound: -0.0532856, upper bound: 0.0532912

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -2.8357103, -1.0795503, -2.8358440, -1.0794785, -0.9595179, 0.9598939
1: -3.5299926, -1.3308620, -3.5303915, -1.3299775, -1.4446702, 1.4444747
2: -1.7500756, -1.0805469, -1.7510450, -1.0799285, -0.2829769, 0.2832703
3: -1.3795612, -0.3874531, -1.3847188, -0.3863657, -0.4420803, 0.4460378
4: -0.9537059, -0.2985285, -0.9538902, -0.2977030, -0.3842809, 0.3836646
5: -0.9963269, 0.0340807, -1.0001625, 0.0348256, -0.4084162, 0.4113993
6: -2.4371753, -1.4726068, -2.4373798, -1.4722620, -0.3692099, 0.3690864
7: 0.8529292, 1.2677653, 0.8523700, 1.2704083, -0.1572573, 0.1551036
8: -3.9803312, -2.4791951, -3.9828892, -2.4785936, -0.7354514, 0.7374392
9: -4.4657326, -2.7495503, -4.4681792, -2.7490089, -1.2561485, 1.2580351

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 382
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 375
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 3507
type: B, layer: 1, pos: 265
type: B, layer: 1, pos: 363
type: B, layer: 1, pos: 318
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 3568
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 376
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 428
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 3321
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 3037
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 3485
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 287
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3323
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 3433
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 3300
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 3452
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 413
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 3303
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 3561
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2961
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 3576
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 3580
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 3578
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 3335
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2434
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 2284
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2335
type: B, layer: 1, pos: 3352
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2494
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 2336
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 3010
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 3574
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 493
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3575
type: B, layer: 1, pos: 3358
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 436
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2462
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2469
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3239
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3362
type: B, layer: 1, pos: 3365
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3569
type: B, layer: 1, pos: 3585

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 382

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0530631, upper bound: 0.0531029
time: 92.75 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0530610, upper bound: 0.0532822
time: 19.23 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -2.8359463, -1.0793564, -2.8359466, -1.0793560, -0.9596437, 0.9606695
1: -3.5323410, -1.3292522, -3.5323608, -1.3292513, -1.4484491, 1.4481106
2: -1.7521778, -1.0798638, -1.7521780, -1.0798638, -0.2842654, 0.2850730
3: -1.3847564, -0.3777312, -1.3847566, -0.3777283, -0.4556412, 0.4431677
4: -0.9553958, -0.2971802, -0.9553963, -0.2971799, -0.3857710, 0.3860729
5: -1.0003088, 0.0410390, -1.0003088, 0.0410399, -0.4184288, 0.4096352
6: -2.4380589, -1.4722542, -2.4380589, -1.4722539, -0.3690820, 0.3700586
7: 0.8477008, 1.2704567, 0.8476999, 1.2704570, -0.1549857, 0.1624571
8: -3.9828966, -2.4737809, -3.9828966, -2.4737787, -0.7427993, 0.7351838
9: -4.4692979, -2.7444847, -4.4692979, -2.7444828, -1.2637721, 1.2621878

Time for backsubstitution: 5.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 382
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 375
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 3507
type: B, layer: 1, pos: 265
type: B, layer: 1, pos: 363
type: B, layer: 1, pos: 318
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 3568
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 376
type: B, layer: 1, pos: 428
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 3321
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 3037
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 3485
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 287
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3323
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 3433
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 3300
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 3452
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 413
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 3303
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 3561
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2961
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 3576
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 3580
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 3578
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 3335
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2434
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 2284
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2335
type: B, layer: 1, pos: 3352
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2494
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 2336
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 3010
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 3574
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 493
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3575
type: B, layer: 1, pos: 3358
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 436
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2462
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2469
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3239
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3362
type: B, layer: 1, pos: 3365
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3569
type: B, layer: 1, pos: 3585

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 382

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532932, upper bound: 0.0531016
time: 138.43 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532850, upper bound: 0.0532930
time: 29.59 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 174.05 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 174.05
Output dim: 7, lower bound: -0.0530631, upper bound: 0.0531029
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 174.05
Output dim: 7, lower bound: -0.0530610, upper bound: 0.0532822
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 174.05
Output dim: 7, lower bound: -0.0532932, upper bound: 0.0531016
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 174.05
Output dim: 7, lower bound: -0.0532850, upper bound: 0.0532930

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -2.8357103, -1.0795548, -2.8358440, -1.0794828, -0.9583546, 0.9598207
1: -3.5299919, -1.3308668, -3.5303910, -1.3299818, -1.4441280, 1.4463834
2: -1.7500751, -1.0805469, -1.7510449, -1.0799284, -0.2829759, 0.2797284
3: -1.3795612, -0.3874563, -1.3847188, -0.3863695, -0.4371840, 0.4460196
4: -0.9536990, -0.2985313, -0.9538824, -0.2977062, -0.3841260, 0.3752446
5: -0.9963257, 0.0340753, -1.0001612, 0.0348192, -0.4023346, 0.4113915
6: -2.4371543, -1.4726073, -2.4373579, -1.4722623, -0.3690603, 0.3684765
7: 0.8529312, 1.2677609, 0.8523722, 1.2704034, -0.1572469, 0.1490463
8: -3.9803307, -2.4792128, -3.9828889, -2.4786141, -0.7221050, 0.7373574
9: -4.4657292, -2.7495561, -4.4681768, -2.7490153, -1.2558352, 1.2591183

Time for backsubstitution: 5.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 375
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 411
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 3507
type: A, layer: 1, pos: 265
type: A, layer: 1, pos: 363
type: A, layer: 1, pos: 318
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 3568
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 376
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 382
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 3321
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 385
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 425
type: A, layer: 1, pos: 3037
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 3485
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 287
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 3433
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 3300
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3452
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 413
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 3303
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 3561
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 3576
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 3580
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 3578
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 3335
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 461
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2434
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 192
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 3559
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2335
type: A, layer: 1, pos: 3352
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 2336
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 3010
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 3574
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 493
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3575
type: A, layer: 1, pos: 3358
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 3557
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 436
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2462
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2469
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3239
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3362
type: A, layer: 1, pos: 3365
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3569
type: A, layer: 1, pos: 3585

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 426

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0528686, upper bound: 0.0532754
time: 168.89 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0530592, upper bound: 0.0532852
time: 12.53 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -2.8357606, -1.0808043, -2.8355134, -1.0811915, -0.9578135, 0.9588696
1: -3.5318673, -1.3308821, -3.5297012, -1.3311229, -1.4462135, 1.4420502
2: -1.7492430, -1.0800085, -1.7488959, -1.0810409, -0.2802533, 0.2816695
3: -1.3846765, -0.3824097, -1.3842257, -0.3829010, -0.4503934, 0.4381082
4: -0.9500729, -0.2972183, -0.9493175, -0.2991160, -0.3778873, 0.3799374
5: -1.0001336, 0.0365847, -0.9998740, 0.0360945, -0.4129211, 0.4036972
6: -2.4376965, -1.4722972, -2.4376447, -1.4723935, -0.3682180, 0.3695487
7: 0.8525722, 1.2704452, 0.8531869, 1.2687864, -0.1493245, 0.1569239
8: -3.9828877, -2.4827719, -3.9796751, -2.4840488, -0.7325149, 0.7227512
9: -4.4687452, -2.7467036, -4.4679747, -2.7470174, -1.2606769, 1.2573612

Time for backsubstitution: 6.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 375
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 411
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 3507
type: A, layer: 1, pos: 265
type: A, layer: 1, pos: 363
type: A, layer: 1, pos: 318
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 3568
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 376
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 3321
type: A, layer: 1, pos: 382
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 385
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 425
type: A, layer: 1, pos: 3037
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 3485
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 287
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 3433
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 3300
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3452
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 413
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 3303
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 3561
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 3576
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 3580
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 3578
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 3335
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 461
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2434
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 192
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 3559
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2335
type: A, layer: 1, pos: 3352
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 2336
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 3010
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 3574
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 493
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3575
type: A, layer: 1, pos: 3358
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 3557
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 436
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2462
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2469
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3239
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3362
type: A, layer: 1, pos: 3365
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3569
type: A, layer: 1, pos: 3585

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 426

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0530962, upper bound: 0.0531161
time: 13.05 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532899, upper bound: 0.0531096
time: 23.93 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -2.8359456, -1.0793600, -2.8359461, -1.0793605, -0.9584807, 0.9605969
1: -3.5323410, -1.3292570, -3.5323606, -1.3292561, -1.4479072, 1.4500201
2: -1.7521772, -1.0798639, -1.7521777, -1.0798639, -0.2842642, 0.2815324
3: -1.3847562, -0.3777347, -1.3847563, -0.3777323, -0.4507436, 0.4431493
4: -0.9553889, -0.2971832, -0.9553884, -0.2971833, -0.3856162, 0.3776530
5: -1.0003076, 0.0410334, -1.0003072, 0.0410337, -0.4123496, 0.4096274
6: -2.4380379, -1.4722542, -2.4380372, -1.4722542, -0.3689325, 0.3694487
7: 0.8477029, 1.2704526, 0.8477023, 1.2704520, -0.1549752, 0.1564001
8: -3.9828961, -2.4737990, -3.9828959, -2.4737997, -0.7294531, 0.7351021
9: -4.4692950, -2.7444906, -4.4692945, -2.7444897, -1.2634585, 1.2632707

Time for backsubstitution: 6.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 375
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 411
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 3507
type: A, layer: 1, pos: 265
type: A, layer: 1, pos: 363
type: A, layer: 1, pos: 318
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 3568
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 376
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 382
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 3321
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 385
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 425
type: A, layer: 1, pos: 3037
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 3485
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 287
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 3433
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 3300
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3452
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 413
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 3303
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 3561
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 3576
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 3580
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 3578
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 3335
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 461
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2434
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 192
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 3559
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2335
type: A, layer: 1, pos: 3352
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 2336
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 3010
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 3574
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 493
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3575
type: A, layer: 1, pos: 3358
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 3557
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 436
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2462
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2469
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3239
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3362
type: A, layer: 1, pos: 3365
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3569
type: A, layer: 1, pos: 3585

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 426

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0530966, upper bound: 0.0532927
time: 80.40 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532857, upper bound: 0.0532957
time: 12.71 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 99.23 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 99.23
Output dim: 7, lower bound: -0.0528686, upper bound: 0.0532754
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 99.23
Output dim: 7, lower bound: -0.0530592, upper bound: 0.0532852
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 99.23
Output dim: 7, lower bound: -0.0530962, upper bound: 0.0531161
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 99.23
Output dim: 7, lower bound: -0.0532899, upper bound: 0.0531096
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 99.23
Output dim: 7, lower bound: -0.0530966, upper bound: 0.0532927
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 99.23
Output dim: 7, lower bound: -0.0532857, upper bound: 0.0532957

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2.8343358, -1.0795636, -2.8346367, -1.0794916, -0.9568963, 0.9584875
1: -3.5288415, -1.3308668, -3.5293519, -1.3299828, -1.4428892, 1.4452279
2: -1.7491170, -1.0805686, -1.7502155, -1.0799475, -0.2820020, 0.2788808
3: -1.3795589, -0.3904009, -1.3847163, -0.3889226, -0.4345413, 0.4430519
4: -0.9526588, -0.2986882, -0.9529690, -0.2978452, -0.3830507, 0.3742920
5: -0.9963004, 0.0323697, -1.0001383, 0.0333400, -0.4008406, 0.4096788
6: -2.4356966, -1.4726157, -2.4360943, -1.4722701, -0.3675916, 0.3671764
7: 0.8536843, 1.2675800, 0.8530251, 1.2702467, -0.1564746, 0.1483751
8: -3.9803107, -2.4802802, -3.9828722, -2.4795389, -0.7211592, 0.7362800
9: -4.4654016, -2.7508349, -4.4678869, -2.7501233, -1.2545364, 1.2576925

Time for backsubstitution: 6.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 375
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 3507
type: B, layer: 1, pos: 265
type: B, layer: 1, pos: 363
type: B, layer: 1, pos: 318
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 3568
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 376
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 428
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 3321
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 3037
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 3485
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 287
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3323
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 3433
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 3300
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 3452
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 413
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 3303
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 3561
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 2961
type: B, layer: 1, pos: 3576
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 3580
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 3578
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 3335
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2434
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 2284
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 2335
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 3352
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2494
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 2336
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 3010
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 3574
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 493
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3575
type: B, layer: 1, pos: 3358
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 436
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2462
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2469
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3239
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3362
type: B, layer: 1, pos: 3365
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3569
type: B, layer: 1, pos: 3585

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3567

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0528740, upper bound: 0.0532286
time: 57.22 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0528681, upper bound: 0.0532802
time: 20.31 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2.8352544, -1.0751388, -2.8354411, -1.0794842, -0.9574426, 0.9634017
1: -3.5296865, -1.3291965, -3.5298436, -1.3299818, -1.4434898, 1.4468629
2: -1.7507014, -1.0773841, -1.7510352, -1.0799297, -0.2827789, 0.2828636
3: -1.3891052, -0.3875623, -1.3847185, -0.3864738, -0.4462019, 0.4434765
4: -0.9537017, -0.2951545, -0.9538672, -0.2977086, -0.3835110, 0.3786355
5: -1.0020819, 0.0340237, -1.0001606, 0.0347739, -0.4079254, 0.4099640
6: -2.4371490, -1.4676762, -2.4372725, -1.4722630, -0.3678113, 0.3730689
7: 0.8529245, 1.2742121, 0.8523839, 1.2704004, -0.1565920, 0.1569890
8: -3.9840603, -2.4792295, -3.9828885, -2.4786439, -0.7257130, 0.7364182
9: -4.4699783, -2.7496080, -4.4681711, -2.7490606, -1.2600849, 1.2585685

Time for backsubstitution: 6.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 375
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 3507
type: B, layer: 1, pos: 265
type: B, layer: 1, pos: 363
type: B, layer: 1, pos: 318
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 3568
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 376
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 428
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 3321
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 3037
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 3485
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 287
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3323
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 3433
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 3300
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 3452
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 413
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 3303
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 3561
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2961
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 3576
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 3580
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 3578
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 3335
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2434
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 2284
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2335
type: B, layer: 1, pos: 3352
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2494
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 2336
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 3010
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 3574
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 493
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3575
type: B, layer: 1, pos: 3358
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 436
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2462
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2469
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3239
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3362
type: B, layer: 1, pos: 3365
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3569
type: B, layer: 1, pos: 3585

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3567

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0530545, upper bound: 0.0532230
time: 99.03 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0530624, upper bound: 0.0532865
time: 12.94 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.8353055, -1.0763881, -2.8351097, -1.0811923, -0.9569017, 0.9624513
1: -3.5315599, -1.3292208, -3.5291545, -1.3311234, -1.4455739, 1.4425266
2: -1.7498572, -1.0768466, -1.7488863, -1.0810418, -0.2800531, 0.2848043
3: -1.3942267, -0.3825152, -1.3842254, -0.3830050, -0.4594254, 0.4355649
4: -0.9500753, -0.2938460, -0.9493018, -0.2991183, -0.3772723, 0.3833266
5: -1.0058953, 0.0365333, -0.9998738, 0.0360489, -0.4185204, 0.4022696
6: -2.4376888, -1.4673661, -2.4375596, -1.4723939, -0.3669683, 0.3741411
7: 0.8525651, 1.2769010, 0.8531983, 1.2687835, -0.1486699, 0.1648256
8: -3.9866185, -2.4827881, -3.9796748, -2.4840784, -0.7361239, 0.7218119
9: -4.4729872, -2.7467556, -4.4679689, -2.7470629, -1.2649230, 1.2568113

Time for backsubstitution: 6.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 375
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 3507
type: B, layer: 1, pos: 265
type: B, layer: 1, pos: 363
type: B, layer: 1, pos: 318
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 3568
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 376
type: B, layer: 1, pos: 428
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 3321
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 3037
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3485
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 287
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3323
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 3433
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 3300
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 3452
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 413
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 3303
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 3561
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2961
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 3576
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 3580
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 3578
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 3335
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2434
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 2284
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2335
type: B, layer: 1, pos: 3352
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2494
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 2336
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 3010
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 3574
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 493
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3575
type: B, layer: 1, pos: 3358
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 436
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2462
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2469
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3239
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3362
type: B, layer: 1, pos: 3365
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3569
type: B, layer: 1, pos: 3585

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3567

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532878, upper bound: 0.0530526
time: 24.79 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532887, upper bound: 0.0531121
time: 61.94 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.8345714, -1.0793698, -2.8347392, -1.0793693, -0.9570220, 0.9592636
1: -3.5311897, -1.3292584, -3.5313210, -1.3292580, -1.4466692, 1.4488649
2: -1.7512205, -1.0798855, -1.7513485, -1.0798826, -0.2832915, 0.2806842
3: -1.3847537, -0.3806798, -1.3847539, -0.3802853, -0.4481012, 0.4401807
4: -0.9543487, -0.2973385, -0.9544749, -0.2973214, -0.3845415, 0.3767010
5: -1.0002824, 0.0393280, -1.0002849, 0.0395544, -0.4108554, 0.4079149
6: -2.4365802, -1.4722632, -2.4367733, -1.4722618, -0.3674636, 0.3681484
7: 0.8484559, 1.2702721, 0.8483550, 1.2702956, -0.1542030, 0.1557289
8: -3.9828777, -2.4748662, -3.9828796, -2.4747241, -0.7285072, 0.7340246
9: -4.4689722, -2.7457700, -4.4690094, -2.7455978, -1.2621623, 1.2618469

Time for backsubstitution: 6.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 375
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 3507
type: B, layer: 1, pos: 265
type: B, layer: 1, pos: 363
type: B, layer: 1, pos: 318
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 3568
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 376
type: B, layer: 1, pos: 428
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 3321
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 3037
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 3485
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 287
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3323
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 3433
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 3300
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 3452
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 413
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 3303
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 3561
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 2961
type: B, layer: 1, pos: 3576
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 3580
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 3578
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 3335
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2434
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 2284
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 2335
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 3352
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2494
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 2336
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 3010
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 3574
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 493
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3575
type: B, layer: 1, pos: 3358
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 436
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2462
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2469
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3239
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3362
type: B, layer: 1, pos: 3365
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3569
type: B, layer: 1, pos: 3585

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3567

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0530976, upper bound: 0.0532376
time: 158.43 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0530890, upper bound: 0.0532890
time: 14.75 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.8354897, -1.0749440, -2.8355432, -1.0793617, -0.9575684, 0.9641778
1: -3.5320387, -1.3275962, -3.5318131, -1.3292565, -1.4472727, 1.4504968
2: -1.7527914, -1.0767016, -1.7521682, -1.0798649, -0.2840635, 0.2846667
3: -1.3943064, -0.3778399, -1.3847560, -0.3778362, -0.4597739, 0.4406058
4: -0.9553913, -0.2938105, -0.9553730, -0.2971856, -0.3850013, 0.3810427
5: -1.0060687, 0.0409821, -1.0003070, 0.0409883, -0.4179479, 0.4081997
6: -2.4380307, -1.4673225, -2.4379513, -1.4722545, -0.3676826, 0.3740411
7: 0.8476960, 1.2769082, 0.8477139, 1.2704492, -0.1543205, 0.1643435
8: -3.9866266, -2.4738162, -3.9828959, -2.4738293, -0.7330621, 0.7341627
9: -4.4735360, -2.7445431, -4.4692883, -2.7445352, -1.2677047, 1.2627208

Time for backsubstitution: 6.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 375
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 3507
type: B, layer: 1, pos: 265
type: B, layer: 1, pos: 363
type: B, layer: 1, pos: 318
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 626
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 627
type: B, layer: 1, pos: 3568
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 628
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 376
type: B, layer: 1, pos: 428
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 576
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 3321
type: B, layer: 1, pos: 574
type: B, layer: 1, pos: 3109
type: B, layer: 1, pos: 642
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 3037
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 2661
type: B, layer: 1, pos: 2225
type: B, layer: 1, pos: 3485
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 287
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 3356
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3323
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2227
type: B, layer: 1, pos: 3433
type: B, layer: 1, pos: 575
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 3324
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 561
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 3300
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 643
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 3452
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 413
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 3303
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 3561
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 2229
type: B, layer: 1, pos: 2330
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2331
type: B, layer: 1, pos: 2529
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 641
type: B, layer: 1, pos: 2617
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3295
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 783
type: B, layer: 1, pos: 2961
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 3576
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 3004
type: B, layer: 1, pos: 3580
type: B, layer: 1, pos: 870
type: B, layer: 1, pos: 2329
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 2962
type: B, layer: 1, pos: 2996
type: B, layer: 1, pos: 3578
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 3335
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2332
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 653
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 2434
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 2284
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 645
type: B, layer: 1, pos: 3559
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2335
type: B, layer: 1, pos: 3352
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 3023
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2494
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 2336
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 3010
type: B, layer: 1, pos: 2306
type: B, layer: 1, pos: 2261
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 696
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 732
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 2312
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2313
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2038
type: B, layer: 1, pos: 2997
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 3253
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 637
type: B, layer: 1, pos: 3574
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 709
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 706
type: B, layer: 1, pos: 3027
type: B, layer: 1, pos: 720
type: B, layer: 1, pos: 493
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 676
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 3575
type: B, layer: 1, pos: 3358
type: B, layer: 1, pos: 721
type: B, layer: 1, pos: 73
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 3348
type: B, layer: 1, pos: 688
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 710
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 434
type: B, layer: 1, pos: 436
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 569
type: B, layer: 1, pos: 666
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 890
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 895
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2462
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2468
type: B, layer: 1, pos: 2469
type: B, layer: 1, pos: 2686
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3239
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3362
type: B, layer: 1, pos: 3365
type: B, layer: 1, pos: 3367
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3494
type: B, layer: 1, pos: 3569
type: B, layer: 1, pos: 3585

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3567

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532827, upper bound: 0.0532312
time: 243.52 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532799, upper bound: 0.0532977
time: 17.03 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 266.72 seconds
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 266.72
Output dim: 7, lower bound: -0.0528740, upper bound: 0.0532286
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 266.72
Output dim: 7, lower bound: -0.0528681, upper bound: 0.0532802
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 266.72
Output dim: 7, lower bound: -0.0530545, upper bound: 0.0532230
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 266.72
Output dim: 7, lower bound: -0.0530624, upper bound: 0.0532865
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 266.72
Output dim: 7, lower bound: -0.0532878, upper bound: 0.0530526
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 266.72
Output dim: 7, lower bound: -0.0532887, upper bound: 0.0531121
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 266.72
Output dim: 7, lower bound: -0.0530976, upper bound: 0.0532376
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 266.72
Output dim: 7, lower bound: -0.0530890, upper bound: 0.0532890
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 266.72
Output dim: 7, lower bound: -0.0532827, upper bound: 0.0532312
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 266.72
Output dim: 7, lower bound: -0.0532799, upper bound: 0.0532977

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2.8343172, -1.0795703, -2.8346162, -1.0794992, -0.9567227, 0.9598619
1: -3.5288408, -1.3308773, -3.5293512, -1.3299932, -1.4427583, 1.4455402
2: -1.7491133, -1.0805686, -1.7502115, -1.0799475, -0.2822440, 0.2787797
3: -1.3795590, -0.3904138, -1.3847165, -0.3889366, -0.4314662, 0.4425368
4: -0.9526546, -0.2986882, -0.9529642, -0.2978451, -0.3830462, 0.3689953
5: -0.9963005, 0.0323640, -1.0001384, 0.0333338, -0.3980441, 0.4096780
6: -2.4356966, -1.4726169, -2.4360943, -1.4722711, -0.3641616, 0.3671752
7: 0.8536883, 1.2675796, 0.8530293, 1.2702460, -0.1564303, 0.1464296
8: -3.9803109, -2.4802871, -3.9828725, -2.4795463, -0.7205404, 0.7361850
9: -4.4654016, -2.7508576, -4.4678879, -2.7501469, -1.2524574, 1.2576822

Time for backsubstitution: 6.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 375
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 411
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 3507
type: A, layer: 1, pos: 265
type: A, layer: 1, pos: 363
type: A, layer: 1, pos: 318
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 626
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 627
type: A, layer: 1, pos: 3568
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 628
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 376
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 382
type: A, layer: 1, pos: 576
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 3321
type: A, layer: 1, pos: 574
type: A, layer: 1, pos: 3109
type: A, layer: 1, pos: 642
type: A, layer: 1, pos: 138
type: A, layer: 1, pos: 385
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 425
type: A, layer: 1, pos: 3037
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 2661
type: A, layer: 1, pos: 2225
type: A, layer: 1, pos: 3485
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 287
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 3356
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3323
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2227
type: A, layer: 1, pos: 3433
type: A, layer: 1, pos: 575
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 3324
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 561
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 3300
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 643
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3452
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 413
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 3303
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 3561
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 2229
type: A, layer: 1, pos: 2330
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2331
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 2529
type: A, layer: 1, pos: 641
type: A, layer: 1, pos: 2617
type: A, layer: 1, pos: 3295
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 783
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 2961
type: A, layer: 1, pos: 3576
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 3004
type: A, layer: 1, pos: 3580
type: A, layer: 1, pos: 870
type: A, layer: 1, pos: 2329
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 2962
type: A, layer: 1, pos: 2996
type: A, layer: 1, pos: 3578
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 3335
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2332
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 653
type: A, layer: 1, pos: 461
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 2434
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 2284
type: A, layer: 1, pos: 192
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 645
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 3559
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2335
type: A, layer: 1, pos: 3352
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2494
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 2336
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 3010
type: A, layer: 1, pos: 2306
type: A, layer: 1, pos: 2261
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 696
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 732
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 2312
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2313
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2038
type: A, layer: 1, pos: 2997
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 3253
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 637
type: A, layer: 1, pos: 3574
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 709
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 706
type: A, layer: 1, pos: 3027
type: A, layer: 1, pos: 720
type: A, layer: 1, pos: 493
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 676
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 3575
type: A, layer: 1, pos: 3358
type: A, layer: 1, pos: 721
type: A, layer: 1, pos: 73
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 3348
type: A, layer: 1, pos: 688
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 3557
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 434
type: A, layer: 1, pos: 436
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 569
type: A, layer: 1, pos: 666
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 890
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 895
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2462
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2468
type: A, layer: 1, pos: 2469
type: A, layer: 1, pos: 2686
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3239
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3362
type: A, layer: 1, pos: 3365
type: A, layer: 1, pos: 3367
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3494
type: A, layer: 1, pos: 3569
type: A, layer: 1, pos: 3585

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 375

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0528385, upper bound: 0.0531952
time: 18.55 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0528630, upper bound: 0.0532766
time: 201.26 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 135.02 + 1827.11 = 1962.13 seconds
