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
execution time: IAR + RelationalAnalysis = 7.74 + 128.99 = 136.73 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0532849, upper bound: 0.0532896

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3507
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 645
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3568
type: DSZ, layer: 1, pos: 3569
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 3580
type: DSZ, layer: 1, pos: 3585

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3507

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532857, upper bound: 0.0532383
time: 316.49 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532330, upper bound: 0.0532960
time: 12.07 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 328.64 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 328.64
Output dim: 7, lower bound: -0.0532857, upper bound: 0.0532383
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 328.64
Output dim: 7, lower bound: -0.0532330, upper bound: 0.0532960

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -2.8359499, -1.0793538, -2.8359499, -1.0793538, -0.9582018, 0.9582882
1: -3.5325108, -1.3292351, -3.5325108, -1.3292351, -1.4485533, 1.4485672
2: -1.7521802, -1.0798626, -1.7521802, -1.0798626, -0.2837688, 0.2838015
3: -1.3847566, -0.3777040, -1.3847566, -0.3777040, -0.4547334, 0.4547045
4: -0.9554005, -0.2971784, -0.9554005, -0.2971784, -0.3851524, 0.3851752
5: -1.0003092, 0.0410480, -1.0003092, 0.0410480, -0.4182151, 0.4182114
6: -2.4380598, -1.4722536, -2.4380598, -1.4722536, -0.3689754, 0.3689489
7: 0.8476936, 1.2704575, 0.8476936, 1.2704575, -0.1624704, 0.1624745
8: -3.9828970, -2.4737611, -3.9828970, -2.4737611, -0.7428630, 0.7428619
9: -4.4693017, -2.7444663, -4.4693017, -2.7444663, -1.2634860, 1.2634796

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 645
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3568
type: DSZ, layer: 1, pos: 3569
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 3580
type: DSZ, layer: 1, pos: 3585

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3493

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532885, upper bound: 0.0532296
time: 113.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532718, upper bound: 0.0532430
time: 143.75 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -2.8359499, -1.0793538, -2.8359499, -1.0793538, -0.9582882, 0.9582018
1: -3.5325108, -1.3292351, -3.5325108, -1.3292351, -1.4485673, 1.4485533
2: -1.7521802, -1.0798626, -1.7521802, -1.0798626, -0.2838016, 0.2837687
3: -1.3847566, -0.3777040, -1.3847566, -0.3777040, -0.4547045, 0.4547335
4: -0.9554005, -0.2971784, -0.9554005, -0.2971784, -0.3851752, 0.3851524
5: -1.0003092, 0.0410480, -1.0003092, 0.0410480, -0.4182114, 0.4182151
6: -2.4380598, -1.4722536, -2.4380598, -1.4722536, -0.3689489, 0.3689754
7: 0.8476936, 1.2704575, 0.8476936, 1.2704575, -0.1624745, 0.1624704
8: -3.9828970, -2.4737611, -3.9828970, -2.4737611, -0.7428619, 0.7428629
9: -4.4693017, -2.7444663, -4.4693017, -2.7444663, -1.2634796, 1.2634860

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 645
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3568
type: DSZ, layer: 1, pos: 3569
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 3580
type: DSZ, layer: 1, pos: 3585

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3493

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532287, upper bound: 0.0532732
time: 26.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532140, upper bound: 0.0532913
time: 16.65 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 48.86 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 48.86
Output dim: 7, lower bound: -0.0532885, upper bound: 0.0532296
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 48.86
Output dim: 7, lower bound: -0.0532718, upper bound: 0.0532430
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 48.86
Output dim: 7, lower bound: -0.0532287, upper bound: 0.0532732
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 48.86
Output dim: 7, lower bound: -0.0532140, upper bound: 0.0532913

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.8359499, -1.0793538, -2.8359499, -1.0793538, -0.9582094, 0.9582966
1: -3.5325108, -1.3292351, -3.5325108, -1.3292351, -1.4485567, 1.4485711
2: -1.7521802, -1.0798626, -1.7521802, -1.0798626, -0.2837710, 0.2838038
3: -1.3847566, -0.3777040, -1.3847566, -0.3777040, -0.4547436, 0.4547154
4: -0.9554005, -0.2971784, -0.9554005, -0.2971784, -0.3851519, 0.3851747
5: -1.0003092, 0.0410480, -1.0003092, 0.0410480, -0.4181916, 0.4181885
6: -2.4380598, -1.4722536, -2.4380598, -1.4722536, -0.3689650, 0.3689387
7: 0.8476936, 1.2704575, 0.8476936, 1.2704575, -0.1624705, 0.1624745
8: -3.9828970, -2.4737611, -3.9828970, -2.4737611, -0.7428334, 0.7428330
9: -4.4693017, -2.7444663, -4.4693017, -2.7444663, -1.2634907, 1.2634847

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 645
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3568
type: DSZ, layer: 1, pos: 3569
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 3580
type: DSZ, layer: 1, pos: 3585

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3485

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532816, upper bound: 0.0532083
time: 42.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532763, upper bound: 0.0532221
time: 98.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.8359499, -1.0793538, -2.8359499, -1.0793538, -0.9582101, 0.9582959
1: -3.5325108, -1.3292351, -3.5325108, -1.3292351, -1.4485574, 1.4485706
2: -1.7521802, -1.0798626, -1.7521802, -1.0798626, -0.2837710, 0.2838039
3: -1.3847566, -0.3777040, -1.3847566, -0.3777040, -0.4547443, 0.4547146
4: -0.9554005, -0.2971784, -0.9554005, -0.2971784, -0.3851519, 0.3851747
5: -1.0003092, 0.0410480, -1.0003092, 0.0410480, -0.4181922, 0.4181879
6: -2.4380598, -1.4722536, -2.4380598, -1.4722536, -0.3689652, 0.3689386
7: 0.8476936, 1.2704575, 0.8476936, 1.2704575, -0.1624704, 0.1624746
8: -3.9828970, -2.4737611, -3.9828970, -2.4737611, -0.7428340, 0.7428323
9: -4.4693017, -2.7444663, -4.4693017, -2.7444663, -1.2634912, 1.2634842

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 645
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3568
type: DSZ, layer: 1, pos: 3569
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 3580
type: DSZ, layer: 1, pos: 3585

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3485

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532729, upper bound: 0.0532257
time: 32.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532579, upper bound: 0.0532421
time: 23.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.8359499, -1.0793538, -2.8359499, -1.0793538, -0.9582959, 0.9582101
1: -3.5325108, -1.3292351, -3.5325108, -1.3292351, -1.4485705, 1.4485573
2: -1.7521802, -1.0798626, -1.7521802, -1.0798626, -0.2838038, 0.2837710
3: -1.3847566, -0.3777040, -1.3847566, -0.3777040, -0.4547147, 0.4547443
4: -0.9554005, -0.2971784, -0.9554005, -0.2971784, -0.3851747, 0.3851519
5: -1.0003092, 0.0410480, -1.0003092, 0.0410480, -0.4181879, 0.4181922
6: -2.4380598, -1.4722536, -2.4380598, -1.4722536, -0.3689386, 0.3689652
7: 0.8476936, 1.2704575, 0.8476936, 1.2704575, -0.1624746, 0.1624704
8: -3.9828970, -2.4737611, -3.9828970, -2.4737611, -0.7428323, 0.7428341
9: -4.4693017, -2.7444663, -4.4693017, -2.7444663, -1.2634842, 1.2634912

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 645
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3568
type: DSZ, layer: 1, pos: 3569
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 3580
type: DSZ, layer: 1, pos: 3585

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3485

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532404, upper bound: 0.0532675
time: 15.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532238, upper bound: 0.0532722
time: 145.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.8359499, -1.0793538, -2.8359499, -1.0793538, -0.9582967, 0.9582095
1: -3.5325108, -1.3292351, -3.5325108, -1.3292351, -1.4485712, 1.4485568
2: -1.7521802, -1.0798626, -1.7521802, -1.0798626, -0.2838038, 0.2837711
3: -1.3847566, -0.3777040, -1.3847566, -0.3777040, -0.4547154, 0.4547435
4: -0.9554005, -0.2971784, -0.9554005, -0.2971784, -0.3851746, 0.3851519
5: -1.0003092, 0.0410480, -1.0003092, 0.0410480, -0.4181885, 0.4181916
6: -2.4380598, -1.4722536, -2.4380598, -1.4722536, -0.3689387, 0.3689651
7: 0.8476936, 1.2704575, 0.8476936, 1.2704575, -0.1624745, 0.1624705
8: -3.9828970, -2.4737611, -3.9828970, -2.4737611, -0.7428330, 0.7428334
9: -4.4693017, -2.7444663, -4.4693017, -2.7444663, -1.2634847, 1.2634907

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3485
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 645
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3568
type: DSZ, layer: 1, pos: 3569
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 3580
type: DSZ, layer: 1, pos: 3585

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3485

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0532218, upper bound: 0.0532078
time: 119.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532034, upper bound: 0.0532733
time: 157.27 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 282.85 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 282.85
Output dim: 7, lower bound: -0.0532816, upper bound: 0.0532083
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 282.85
Output dim: 7, lower bound: -0.0532763, upper bound: 0.0532221
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 282.85
Output dim: 7, lower bound: -0.0532729, upper bound: 0.0532257
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 282.85
Output dim: 7, lower bound: -0.0532579, upper bound: 0.0532421
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 282.85
Output dim: 7, lower bound: -0.0532404, upper bound: 0.0532675
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 282.85
Output dim: 7, lower bound: -0.0532238, upper bound: 0.0532722
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 282.85
Output dim: 7, lower bound: -0.0532218, upper bound: 0.0532078
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 282.85
Output dim: 7, lower bound: -0.0532034, upper bound: 0.0532733

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.8359499, -1.0793538, -2.8359499, -1.0793538, -0.9583313, 0.9584080
1: -3.5325108, -1.3292351, -3.5325108, -1.3292351, -1.4486063, 1.4486187
2: -1.7521802, -1.0798626, -1.7521802, -1.0798626, -0.2824741, 0.2824652
3: -1.3847566, -0.3777040, -1.3847566, -0.3777040, -0.4547679, 0.4547415
4: -0.9554005, -0.2971784, -0.9554005, -0.2971784, -0.3844696, 0.3844905
5: -1.0003092, 0.0410480, -1.0003092, 0.0410480, -0.4182191, 0.4182172
6: -2.4380598, -1.4722536, -2.4380598, -1.4722536, -0.3682698, 0.3682576
7: 0.8476936, 1.2704575, 0.8476936, 1.2704575, -0.1624715, 0.1624873
8: -3.9828970, -2.4737611, -3.9828970, -2.4737611, -0.7428731, 0.7428568
9: -4.4693017, -2.7444663, -4.4693017, -2.7444663, -1.2634342, 1.2634242

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 645
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3568
type: DSZ, layer: 1, pos: 3569
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 3580
type: DSZ, layer: 1, pos: 3585

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3508

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532808, upper bound: 0.0531861
time: 109.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532545, upper bound: 0.0532154
time: 14.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.8359499, -1.0793538, -2.8359499, -1.0793538, -0.9583209, 0.9584185
1: -3.5325108, -1.3292351, -3.5325108, -1.3292351, -1.4486042, 1.4486206
2: -1.7521802, -1.0798626, -1.7521802, -1.0798626, -0.2824326, 0.2825067
3: -1.3847566, -0.3777040, -1.3847566, -0.3777040, -0.4547697, 0.4547397
4: -0.9554005, -0.2971784, -0.9554005, -0.2971784, -0.3844678, 0.3844923
5: -1.0003092, 0.0410480, -1.0003092, 0.0410480, -0.4182203, 0.4182159
6: -2.4380598, -1.4722536, -2.4380598, -1.4722536, -0.3682839, 0.3682435
7: 0.8476936, 1.2704575, 0.8476936, 1.2704575, -0.1624833, 0.1624755
8: -3.9828970, -2.4737611, -3.9828970, -2.4737611, -0.7428572, 0.7428728
9: -4.4693017, -2.7444663, -4.4693017, -2.7444663, -1.2634301, 1.2634282

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 645
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3568
type: DSZ, layer: 1, pos: 3569
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 3580
type: DSZ, layer: 1, pos: 3585

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3508

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532770, upper bound: 0.0531985
time: 10.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532478, upper bound: 0.0532300
time: 26.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.8359499, -1.0793538, -2.8359499, -1.0793538, -0.9583321, 0.9584073
1: -3.5325108, -1.3292351, -3.5325108, -1.3292351, -1.4486070, 1.4486179
2: -1.7521802, -1.0798626, -1.7521802, -1.0798626, -0.2824740, 0.2824653
3: -1.3847566, -0.3777040, -1.3847566, -0.3777040, -0.4547687, 0.4547407
4: -0.9554005, -0.2971784, -0.9554005, -0.2971784, -0.3844696, 0.3844905
5: -1.0003092, 0.0410480, -1.0003092, 0.0410480, -0.4182196, 0.4182166
6: -2.4380598, -1.4722536, -2.4380598, -1.4722536, -0.3682700, 0.3682574
7: 0.8476936, 1.2704575, 0.8476936, 1.2704575, -0.1624714, 0.1624874
8: -3.9828970, -2.4737611, -3.9828970, -2.4737611, -0.7428738, 0.7428561
9: -4.4693017, -2.7444663, -4.4693017, -2.7444663, -1.2634346, 1.2634237

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 645
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3568
type: DSZ, layer: 1, pos: 3569
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 3580
type: DSZ, layer: 1, pos: 3585

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3508

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532666, upper bound: 0.0531973
time: 93.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532427, upper bound: 0.0532219
time: 60.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.8359499, -1.0793538, -2.8359499, -1.0793538, -0.9583216, 0.9584178
1: -3.5325108, -1.3292351, -3.5325108, -1.3292351, -1.4486048, 1.4486201
2: -1.7521802, -1.0798626, -1.7521802, -1.0798626, -0.2824325, 0.2825068
3: -1.3847566, -0.3777040, -1.3847566, -0.3777040, -0.4547704, 0.4547390
4: -0.9554005, -0.2971784, -0.9554005, -0.2971784, -0.3844678, 0.3844923
5: -1.0003092, 0.0410480, -1.0003092, 0.0410480, -0.4182208, 0.4182154
6: -2.4380598, -1.4722536, -2.4380598, -1.4722536, -0.3682841, 0.3682433
7: 0.8476936, 1.2704575, 0.8476936, 1.2704575, -0.1624832, 0.1624756
8: -3.9828970, -2.4737611, -3.9828970, -2.4737611, -0.7428579, 0.7428721
9: -4.4693017, -2.7444663, -4.4693017, -2.7444663, -1.2634306, 1.2634277

Time for backsubstitution: 5.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 561
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 576
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 575
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3295
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 3027
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2313
type: DSZ, layer: 1, pos: 3290
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 721
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 720
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3303
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 706
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 318
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 574
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 3575
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 126
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 265
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 287
type: DSZ, layer: 1, pos: 3300
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 15
type: DSZ, layer: 1, pos: 2529
type: DSZ, layer: 1, pos: 2514
type: DSZ, layer: 1, pos: 3253
type: DSZ, layer: 1, pos: 783
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2335
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2336
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3010
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 2434
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 376
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 375
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3109
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3323
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 28
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 428
type: DSZ, layer: 1, pos: 434
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 493
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 569
type: DSZ, layer: 1, pos: 627
type: DSZ, layer: 1, pos: 628
type: DSZ, layer: 1, pos: 637
type: DSZ, layer: 1, pos: 641
type: DSZ, layer: 1, pos: 642
type: DSZ, layer: 1, pos: 643
type: DSZ, layer: 1, pos: 645
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 666
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 688
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 696
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 709
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2284
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2329
type: DSZ, layer: 1, pos: 2330
type: DSZ, layer: 1, pos: 2331
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2462
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2468
type: DSZ, layer: 1, pos: 2469
type: DSZ, layer: 1, pos: 2479
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2494
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2661
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2686
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2961
type: DSZ, layer: 1, pos: 2962
type: DSZ, layer: 1, pos: 2997
type: DSZ, layer: 1, pos: 3004
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3239
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3352
type: DSZ, layer: 1, pos: 3356
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3362
type: DSZ, layer: 1, pos: 3365
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3368
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3494
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3568
type: DSZ, layer: 1, pos: 3569
type: DSZ, layer: 1, pos: 3576
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 3580
type: DSZ, layer: 1, pos: 3585

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3508

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0532553, upper bound: 0.0532134
time: 17.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0532290, upper bound: 0.0532358
time: 96.33 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 119.70 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 119.70
Output dim: 7, lower bound: -0.0532808, upper bound: 0.0531861
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 119.70
Output dim: 7, lower bound: -0.0532545, upper bound: 0.0532154
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 119.70
Output dim: 7, lower bound: -0.0532770, upper bound: 0.0531985
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 119.70
Output dim: 7, lower bound: -0.0532478, upper bound: 0.0532300
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 119.70
Output dim: 7, lower bound: -0.0532666, upper bound: 0.0531973
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 119.70
Output dim: 7, lower bound: -0.0532427, upper bound: 0.0532219
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 119.70
Output dim: 7, lower bound: -0.0532553, upper bound: 0.0532134
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 119.70
Output dim: 7, lower bound: -0.0532290, upper bound: 0.0532358
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 119.70
Output dim: 7, lower bound: -0.0532404, upper bound: 0.0532675
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 119.70
Output dim: 7, lower bound: -0.0532238, upper bound: 0.0532722
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 119.70
Output dim: 7, lower bound: -0.0532034, upper bound: 0.0532733

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 136.73 + 1753.26 = 1889.99 seconds
