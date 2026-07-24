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
execution time: IAR + RelationalAnalysis = 7.94 + 130.36 = 138.30 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0532849, upper bound: 0.0532896

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3569
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 645
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3580
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3568
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 732

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532562, upper bound: 0.0532957
time: 12.32 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532846, upper bound: 0.0532641
time: 121.67 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 134.00 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 134.00
Output dim: 7, lower bound: -0.0532562, upper bound: 0.0532957
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 134.00
Output dim: 7, lower bound: -0.0532846, upper bound: 0.0532641

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -2.8359499, -1.0793538, -2.8359499, -1.0793538, -0.9570399, 0.9570191
1: -3.5325108, -1.3292351, -3.5325108, -1.3292351, -1.4411421, 1.4411433
2: -1.7521802, -1.0798626, -1.7521802, -1.0798626, -0.2849873, 0.2849890
3: -1.3847566, -0.3777040, -1.3847566, -0.3777040, -0.4556091, 0.4556068
4: -0.9554005, -0.2971784, -0.9554005, -0.2971784, -0.3815596, 0.3816730
5: -1.0003092, 0.0410480, -1.0003092, 0.0410480, -0.4182146, 0.4182135
6: -2.4380598, -1.4722536, -2.4380598, -1.4722536, -0.3700747, 0.3700745
7: 0.8476936, 1.2704575, 0.8476936, 1.2704575, -0.1622277, 0.1622312
8: -3.9828970, -2.4737611, -3.9828970, -2.4737611, -0.7346373, 0.7344203
9: -4.4693017, -2.7444663, -4.4693017, -2.7444663, -1.2547388, 1.2547290

Time for backsubstitution: 6.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 3568
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3569
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 3580
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 645
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 861

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2332

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532500, upper bound: 0.0532911
time: 28.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532629, upper bound: 0.0532929
time: 131.36 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -2.8359499, -1.0793538, -2.8359499, -1.0793538, -0.9570192, 0.9570400
1: -3.5325108, -1.3292351, -3.5325108, -1.3292351, -1.4411433, 1.4411421
2: -1.7521802, -1.0798626, -1.7521802, -1.0798626, -0.2849890, 0.2849874
3: -1.3847566, -0.3777040, -1.3847566, -0.3777040, -0.4556068, 0.4556091
4: -0.9554005, -0.2971784, -0.9554005, -0.2971784, -0.3816729, 0.3815597
5: -1.0003092, 0.0410480, -1.0003092, 0.0410480, -0.4182136, 0.4182145
6: -2.4380598, -1.4722536, -2.4380598, -1.4722536, -0.3700745, 0.3700747
7: 0.8476936, 1.2704575, 0.8476936, 1.2704575, -0.1622312, 0.1622277
8: -3.9828970, -2.4737611, -3.9828970, -2.4737611, -0.7344202, 0.7346373
9: -4.4693017, -2.7444663, -4.4693017, -2.7444663, -1.2547290, 1.2547388

Time for backsubstitution: 6.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 3569
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3580
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 3568
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 645
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3348

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 489

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532650, upper bound: 0.0532665
time: 13.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532892, upper bound: 0.0532435
time: 198.82 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 218.64 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 218.64
Output dim: 7, lower bound: -0.0532500, upper bound: 0.0532911
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 218.64
Output dim: 7, lower bound: -0.0532629, upper bound: 0.0532929
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 218.64
Output dim: 7, lower bound: -0.0532650, upper bound: 0.0532665
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 218.64
Output dim: 7, lower bound: -0.0532892, upper bound: 0.0532435

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.8359499, -1.0793538, -2.8359499, -1.0793538, -0.9551120, 0.9548838
1: -3.5325108, -1.3292351, -3.5325108, -1.3292351, -1.4390551, 1.4388039
2: -1.7521802, -1.0798626, -1.7521802, -1.0798626, -0.2849793, 0.2849809
3: -1.3847566, -0.3777040, -1.3847566, -0.3777040, -0.4550582, 0.4551183
4: -0.9554005, -0.2971784, -0.9554005, -0.2971784, -0.3814778, 0.3816012
5: -1.0003092, 0.0410480, -1.0003092, 0.0410480, -0.4174827, 0.4175512
6: -2.4380598, -1.4722536, -2.4380598, -1.4722536, -0.3685118, 0.3686662
7: 0.8476936, 1.2704575, 0.8476936, 1.2704575, -0.1622187, 0.1622225
8: -3.9828970, -2.4737611, -3.9828970, -2.4737611, -0.7339794, 0.7336903
9: -4.4693017, -2.7444663, -4.4693017, -2.7444663, -1.2537969, 1.2536460

Time for backsubstitution: 6.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 645
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3568
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 3569
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3580
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2512

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532370, upper bound: 0.0532797
time: 168.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532491, upper bound: 0.0532677
time: 158.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.8359499, -1.0793538, -2.8359499, -1.0793538, -0.9549047, 0.9550911
1: -3.5325108, -1.3292351, -3.5325108, -1.3292351, -1.4388027, 1.4390564
2: -1.7521802, -1.0798626, -1.7521802, -1.0798626, -0.2849793, 0.2849810
3: -1.3847566, -0.3777040, -1.3847566, -0.3777040, -0.4551206, 0.4550559
4: -0.9554005, -0.2971784, -0.9554005, -0.2971784, -0.3814880, 0.3815911
5: -1.0003092, 0.0410480, -1.0003092, 0.0410480, -0.4175522, 0.4174817
6: -2.4380598, -1.4722536, -2.4380598, -1.4722536, -0.3686663, 0.3685117
7: 0.8476936, 1.2704575, 0.8476936, 1.2704575, -0.1622190, 0.1622222
8: -3.9828970, -2.4737611, -3.9828970, -2.4737611, -0.7339073, 0.7337624
9: -4.4693017, -2.7444663, -4.4693017, -2.7444663, -1.2536559, 1.2537872

Time for backsubstitution: 6.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3568
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 645
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3580
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 3569
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2628

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3568

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532279, upper bound: 0.0532875
time: 79.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532524, upper bound: 0.0532678
time: 208.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.8359499, -1.0793538, -2.8359499, -1.0793538, -0.9570090, 0.9570037
1: -3.5325108, -1.3292351, -3.5325108, -1.3292351, -1.4411321, 1.4411329
2: -1.7521802, -1.0798626, -1.7521802, -1.0798626, -0.2848261, 0.2848445
3: -1.3847566, -0.3777040, -1.3847566, -0.3777040, -0.4555185, 0.4555095
4: -0.9554005, -0.2971784, -0.9554005, -0.2971784, -0.3816572, 0.3815458
5: -1.0003092, 0.0410480, -1.0003092, 0.0410480, -0.4181788, 0.4181996
6: -2.4380598, -1.4722536, -2.4380598, -1.4722536, -0.3700004, 0.3700086
7: 0.8476936, 1.2704575, 0.8476936, 1.2704575, -0.1622221, 0.1622210
8: -3.9828970, -2.4737611, -3.9828970, -2.4737611, -0.7342919, 0.7344836
9: -4.4693017, -2.7444663, -4.4693017, -2.7444663, -1.2547202, 1.2547375

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 3569
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3568
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 645
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3580
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 569

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3010

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532562, upper bound: 0.0532296
time: 47.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532518, upper bound: 0.0532728
time: 10.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.8359499, -1.0793538, -2.8359499, -1.0793538, -0.9569828, 0.9570299
1: -3.5325108, -1.3292351, -3.5325108, -1.3292351, -1.4411343, 1.4411308
2: -1.7521802, -1.0798626, -1.7521802, -1.0798626, -0.2848461, 0.2848245
3: -1.3847566, -0.3777040, -1.3847566, -0.3777040, -0.4555071, 0.4555208
4: -0.9554005, -0.2971784, -0.9554005, -0.2971784, -0.3816592, 0.3815439
5: -1.0003092, 0.0410480, -1.0003092, 0.0410480, -0.4181986, 0.4181798
6: -2.4380598, -1.4722536, -2.4380598, -1.4722536, -0.3700085, 0.3700004
7: 0.8476936, 1.2704575, 0.8476936, 1.2704575, -0.1622245, 0.1622186
8: -3.9828970, -2.4737611, -3.9828970, -2.4737611, -0.7342665, 0.7345089
9: -4.4693017, -2.7444663, -4.4693017, -2.7444663, -1.2547277, 1.2547299

Time for backsubstitution: 6.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3580
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 3568
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 645
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3569
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 3367

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2996

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532799, upper bound: 0.0532455
time: 13.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532800, upper bound: 0.0532374
time: 11.35 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 30.52 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.52
Output dim: 7, lower bound: -0.0532370, upper bound: 0.0532797
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.52
Output dim: 7, lower bound: -0.0532491, upper bound: 0.0532677
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.52
Output dim: 7, lower bound: -0.0532279, upper bound: 0.0532875
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.52
Output dim: 7, lower bound: -0.0532524, upper bound: 0.0532678
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.52
Output dim: 7, lower bound: -0.0532562, upper bound: 0.0532296
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.52
Output dim: 7, lower bound: -0.0532518, upper bound: 0.0532728
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 30.52
Output dim: 7, lower bound: -0.0532799, upper bound: 0.0532455
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 30.52
Output dim: 7, lower bound: -0.0532800, upper bound: 0.0532374

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.8359499, -1.0793538, -2.8359499, -1.0793538, -0.9549718, 0.9547111
1: -3.5325108, -1.3292351, -3.5325108, -1.3292351, -1.4383600, 1.4380898
2: -1.7521802, -1.0798626, -1.7521802, -1.0798626, -0.2849614, 0.2849603
3: -1.3847566, -0.3777040, -1.3847566, -0.3777040, -0.4549033, 0.4549501
4: -0.9554005, -0.2971784, -0.9554005, -0.2971784, -0.3813885, 0.3815530
5: -1.0003092, 0.0410480, -1.0003092, 0.0410480, -0.4174572, 0.4175229
6: -2.4380598, -1.4722536, -2.4380598, -1.4722536, -0.3685006, 0.3686546
7: 0.8476936, 1.2704575, 0.8476936, 1.2704575, -0.1621571, 0.1621624
8: -3.9828970, -2.4737611, -3.9828970, -2.4737611, -0.7327106, 0.7322309
9: -4.4693017, -2.7444663, -4.4693017, -2.7444663, -1.2526042, 1.2524056

Time for backsubstitution: 6.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 3568
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3580
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 3569
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 645

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 819

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532319, upper bound: 0.0532844
time: 13.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532329, upper bound: 0.0532483
time: 209.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.8359499, -1.0793538, -2.8359499, -1.0793538, -0.9549391, 0.9547437
1: -3.5325108, -1.3292351, -3.5325108, -1.3292351, -1.4383409, 1.4381089
2: -1.7521802, -1.0798626, -1.7521802, -1.0798626, -0.2849587, 0.2849630
3: -1.3847566, -0.3777040, -1.3847566, -0.3777040, -0.4548900, 0.4549633
4: -0.9554005, -0.2971784, -0.9554005, -0.2971784, -0.3814295, 0.3815120
5: -1.0003092, 0.0410480, -1.0003092, 0.0410480, -0.4174544, 0.4175257
6: -2.4380598, -1.4722536, -2.4380598, -1.4722536, -0.3685002, 0.3686549
7: 0.8476936, 1.2704575, 0.8476936, 1.2704575, -0.1621586, 0.1621610
8: -3.9828970, -2.4737611, -3.9828970, -2.4737611, -0.7325200, 0.7324215
9: -4.4693017, -2.7444663, -4.4693017, -2.7444663, -1.2525566, 1.2524533

Time for backsubstitution: 6.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3580
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 3569
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 645
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 3568
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 428

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2217

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532397, upper bound: 0.0532714
time: 14.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532457, upper bound: 0.0532636
time: 18.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.8359499, -1.0793538, -2.8359499, -1.0793538, -0.9545929, 0.9548081
1: -3.5325108, -1.3292351, -3.5325108, -1.3292351, -1.4387512, 1.4390081
2: -1.7521802, -1.0798626, -1.7521802, -1.0798626, -0.2849404, 0.2849375
3: -1.3847566, -0.3777040, -1.3847566, -0.3777040, -0.4544438, 0.4543109
4: -0.9554005, -0.2971784, -0.9554005, -0.2971784, -0.3806589, 0.3808402
5: -1.0003092, 0.0410480, -1.0003092, 0.0410480, -0.4171527, 0.4170434
6: -2.4380598, -1.4722536, -2.4380598, -1.4722536, -0.3681807, 0.3679762
7: 0.8476936, 1.2704575, 0.8476936, 1.2704575, -0.1618923, 0.1619264
8: -3.9828970, -2.4737611, -3.9828970, -2.4737611, -0.7337694, 0.7336110
9: -4.4693017, -2.7444663, -4.4693017, -2.7444663, -1.2533627, 1.2534642

Time for backsubstitution: 6.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 645
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 3580
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 3585
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3569
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2485

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532332, upper bound: 0.0532892
time: 168.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532318, upper bound: 0.0532845
time: 191.89 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 366.55 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 366.55
Output dim: 7, lower bound: -0.0532319, upper bound: 0.0532844
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 366.55
Output dim: 7, lower bound: -0.0532329, upper bound: 0.0532483
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 366.55
Output dim: 7, lower bound: -0.0532397, upper bound: 0.0532714
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 366.55
Output dim: 7, lower bound: -0.0532457, upper bound: 0.0532636
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 366.55
Output dim: 7, lower bound: -0.0532332, upper bound: 0.0532892
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 366.55
Output dim: 7, lower bound: -0.0532318, upper bound: 0.0532845
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 366.55
Output dim: 7, lower bound: -0.0532524, upper bound: 0.0532678
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 366.55
Output dim: 7, lower bound: -0.0532562, upper bound: 0.0532296
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 366.55
Output dim: 7, lower bound: -0.0532518, upper bound: 0.0532728
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 366.55
Output dim: 7, lower bound: -0.0532799, upper bound: 0.0532455
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 366.55
Output dim: 7, lower bound: -0.0532800, upper bound: 0.0532374

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 138.30 + 1874.01 = 2012.31 seconds
