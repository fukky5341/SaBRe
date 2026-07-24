## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 8)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.0282733983


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7426962, 0.7426962)
1: (-4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5056798, 1.5056801)
2: (-0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6989986, 0.6989987)
3: (-1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2825801, 0.2825801)
4: (-0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7384548, 0.7384549)
5: (-1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3549864, 0.3549864)
6: (0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1197150, 0.1197150)
7: (-2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7333935, 0.7333934)
8: (-4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9601736, 0.9601736)
9: (-4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0755296, 1.0755298)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.42 + 121.35 = 128.77 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0282980, upper bound: 0.0283015

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 2501

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2188

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282962, upper bound: 0.0282993
time: 148.93 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282982, upper bound: 0.0282987
time: 421.00 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 569.95 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 569.95
Output dim: 6, lower bound: -0.0282962, upper bound: 0.0282993
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 569.95
Output dim: 6, lower bound: -0.0282982, upper bound: 0.0282987

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7423031, 0.7422027
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5051608, 1.5049987
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6989940, 0.6989959
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2825646, 0.2825647
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7383814, 0.7383813
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3549474, 0.3549553
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1197131, 0.1197134
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7332438, 0.7332428
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9592413, 0.9589663
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0748096, 1.0745423

Time for backsubstitution: 5.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2611

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282946, upper bound: 0.0283003
time: 10.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282934, upper bound: 0.0283000
time: 115.42 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7422028, 0.7423031
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5049984, 1.5051608
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6989959, 0.6989942
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2825647, 0.2825646
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7383813, 0.7383814
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3549553, 0.3549473
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1197134, 0.1197131
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7332427, 0.7332438
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9589662, 0.9592414
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0745423, 1.0748098

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3098

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282858, upper bound: 0.0283006
time: 100.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282858, upper bound: 0.0282907
time: 31.87 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 138.84 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 138.84
Output dim: 6, lower bound: -0.0282946, upper bound: 0.0283003
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 138.84
Output dim: 6, lower bound: -0.0282934, upper bound: 0.0283000
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 138.84
Output dim: 6, lower bound: -0.0282858, upper bound: 0.0283006
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 138.84
Output dim: 6, lower bound: -0.0282858, upper bound: 0.0282907

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7409551, 0.7408851
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5037616, 1.5036349
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6990149, 0.6990163
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2824953, 0.2824961
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7383712, 0.7383711
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3549355, 0.3549424
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1196878, 0.1196866
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7324671, 0.7324418
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9557866, 0.9556330
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0720717, 1.0719379

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2601

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2583

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282845, upper bound: 0.0282854
time: 198.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282828, upper bound: 0.0282870
time: 31.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7409855, 0.7408547
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5037972, 1.5035992
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6990144, 0.6990167
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2824960, 0.2824954
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7383713, 0.7383711
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3549345, 0.3549435
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1196862, 0.1196881
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7324430, 0.7324659
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9559082, 0.9555115
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0722053, 1.0718044

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2453

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 320

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282797, upper bound: 0.0282651
time: 99.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282581, upper bound: 0.0282873
time: 7.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7422028, 0.7423031
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5049984, 1.5051608
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6989959, 0.6989942
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2825647, 0.2825646
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7383813, 0.7383814
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3549553, 0.3549473
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1197134, 0.1197131
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7332427, 0.7332438
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9589662, 0.9592414
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0745423, 1.0748098

Time for backsubstitution: 6.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2583

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2570

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282590, upper bound: 0.0282954
time: 157.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282820, upper bound: 0.0282723
time: 184.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7422028, 0.7423031
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5049984, 1.5051608
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6989959, 0.6989942
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2825647, 0.2825646
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7383813, 0.7383814
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3549553, 0.3549473
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1197134, 0.1197131
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7332427, 0.7332438
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9589662, 0.9592414
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0745423, 1.0748098

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3153

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282854, upper bound: 0.0282922
time: 10.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282835, upper bound: 0.0282868
time: 75.94 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 92.75 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 92.75
Output dim: 6, lower bound: -0.0282845, upper bound: 0.0282854
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 92.75
Output dim: 6, lower bound: -0.0282828, upper bound: 0.0282870
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 92.75
Output dim: 6, lower bound: -0.0282797, upper bound: 0.0282651
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 92.75
Output dim: 6, lower bound: -0.0282581, upper bound: 0.0282873
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 92.75
Output dim: 6, lower bound: -0.0282590, upper bound: 0.0282954
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 92.75
Output dim: 6, lower bound: -0.0282820, upper bound: 0.0282723
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 92.75
Output dim: 6, lower bound: -0.0282854, upper bound: 0.0282922
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 92.75
Output dim: 6, lower bound: -0.0282835, upper bound: 0.0282868

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7406304, 0.7405366
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5033947, 1.5032611
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6989942, 0.6990018
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2823910, 0.2823845
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7383664, 0.7383700
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3548075, 0.3548268
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1196841, 0.1196821
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7324483, 0.7324212
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9555146, 0.9553411
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0717191, 1.0715818

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2882

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 873

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282839, upper bound: 0.0282847
time: 43.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282830, upper bound: 0.0282878
time: 10.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7406065, 0.7405605
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5033880, 1.5032678
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6990005, 0.6989955
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2823837, 0.2823918
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7383700, 0.7383665
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3548199, 0.3548144
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1196833, 0.1196829
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7324464, 0.7324231
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9554945, 0.9553611
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0717155, 1.0715852

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 735

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282808, upper bound: 0.0282887
time: 13.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282808, upper bound: 0.0282893
time: 12.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7409409, 0.7405498
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5034404, 1.5031276
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6989008, 0.6988631
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2824657, 0.2823937
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7380974, 0.7381712
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3547858, 0.3545403
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1195675, 0.1195561
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7315428, 0.7309000
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9558193, 0.9554676
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0721036, 1.0717367

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2452

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0282730, upper bound: 0.0282610
time: 33.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282781, upper bound: 0.0282595
time: 37.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7406806, 0.7408102
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5033257, 1.5032420
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6988606, 0.6989031
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2823942, 0.2824651
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7381714, 0.7380972
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3545313, 0.3547948
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1195542, 0.1195694
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7308771, 0.7315657
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9558644, 0.9554228
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0721377, 1.0717028

Time for backsubstitution: 6.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2982

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2431

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282561, upper bound: 0.0282866
time: 10.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282562, upper bound: 0.0282832
time: 38.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7389973, 0.7392120
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5014070, 1.5017939
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6989361, 0.6989294
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2815731, 0.2815376
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7383558, 0.7383568
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3537441, 0.3536869
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1197046, 0.1197048
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7331045, 0.7330964
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9555311, 0.9559946
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0702400, 1.0706915

Time for backsubstitution: 5.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3126

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3139

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282573, upper bound: 0.0283014
time: 11.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282573, upper bound: 0.0282996
time: 13.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7391115, 0.7390978
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5016314, 1.5015693
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6989313, 0.6989343
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2815377, 0.2815730
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7383566, 0.7383559
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3536949, 0.3537362
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1197051, 0.1197043
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7330951, 0.7331058
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9557195, 0.9558063
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0704243, 1.0705074

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 335

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3155

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282825, upper bound: 0.0282702
time: 22.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282831, upper bound: 0.0282717
time: 124.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7421957, 0.7422925
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5049980, 1.5051603
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6989750, 0.6989720
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2825643, 0.2825642
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7383782, 0.7383776
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3549531, 0.3549454
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1197132, 0.1197129
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7332392, 0.7332398
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9589631, 0.9592371
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0745410, 1.0748079

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 308

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 180

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282860, upper bound: 0.0282866
time: 82.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282855, upper bound: 0.0282861
time: 104.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7421921, 0.7422960
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5049977, 1.5051603
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6989738, 0.6989731
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2825643, 0.2825642
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7383776, 0.7383783
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3549534, 0.3549451
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1197133, 0.1197129
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7332386, 0.7332403
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9589621, 0.9592381
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0745406, 1.0748084

Time for backsubstitution: 6.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 855

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3468

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282710, upper bound: 0.0282884
time: 81.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282860, upper bound: 0.0282751
time: 16.17 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 104.27 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 104.27
Output dim: 6, lower bound: -0.0282839, upper bound: 0.0282847
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 104.27
Output dim: 6, lower bound: -0.0282830, upper bound: 0.0282878
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 104.27
Output dim: 6, lower bound: -0.0282808, upper bound: 0.0282887
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 104.27
Output dim: 6, lower bound: -0.0282808, upper bound: 0.0282893
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 104.27
Output dim: 6, lower bound: -0.0282730, upper bound: 0.0282610
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 104.27
Output dim: 6, lower bound: -0.0282781, upper bound: 0.0282595
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 104.27
Output dim: 6, lower bound: -0.0282561, upper bound: 0.0282866
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 104.27
Output dim: 6, lower bound: -0.0282562, upper bound: 0.0282832
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 104.27
Output dim: 6, lower bound: -0.0282573, upper bound: 0.0283014
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 104.27
Output dim: 6, lower bound: -0.0282573, upper bound: 0.0282996
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 104.27
Output dim: 6, lower bound: -0.0282825, upper bound: 0.0282702
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 104.27
Output dim: 6, lower bound: -0.0282831, upper bound: 0.0282717
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 104.27
Output dim: 6, lower bound: -0.0282860, upper bound: 0.0282866
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 104.27
Output dim: 6, lower bound: -0.0282855, upper bound: 0.0282861
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 104.27
Output dim: 6, lower bound: -0.0282710, upper bound: 0.0282884
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 104.27
Output dim: 6, lower bound: -0.0282860, upper bound: 0.0282751

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7405830, 0.7404906
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5032313, 1.5030937
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6989887, 0.6989971
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2823694, 0.2823638
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7382974, 0.7382993
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3548025, 0.3548228
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1196803, 0.1196783
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7323728, 0.7323437
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9551297, 0.9549677
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0712265, 1.0710912

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 252

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3065

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282781, upper bound: 0.0282807
time: 133.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282798, upper bound: 0.0282769
time: 102.05 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7405844, 0.7404892
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5032272, 1.5030980
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6989893, 0.6989965
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2823704, 0.2823628
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7382959, 0.7383010
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3548035, 0.3548219
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1196802, 0.1196784
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7323709, 0.7323456
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9551415, 0.9549563
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0712284, 1.0710893

Time for backsubstitution: 6.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 221

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2696

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282844, upper bound: 0.0282821
time: 100.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282844, upper bound: 0.0282875
time: 11.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7406065, 0.7405605
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5033880, 1.5032678
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6990005, 0.6989955
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2823837, 0.2823918
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7383700, 0.7383665
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3548199, 0.3548144
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1196833, 0.1196829
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7324464, 0.7324231
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9554945, 0.9553611
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0717155, 1.0715852

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 222

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3428

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282828, upper bound: 0.0282827
time: 16.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282752, upper bound: 0.0282870
time: 21.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7406065, 0.7405605
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5033880, 1.5032678
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6990005, 0.6989955
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2823837, 0.2823918
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7383700, 0.7383665
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3548199, 0.3548144
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1196833, 0.1196829
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7324464, 0.7324231
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9554945, 0.9553611
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0717155, 1.0715852

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2220

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2946

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282772, upper bound: 0.0282726
time: 14.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282683, upper bound: 0.0282822
time: 48.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7409372, 0.7405461
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5034397, 1.5031271
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6988993, 0.6988617
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2824640, 0.2823920
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7380974, 0.7381711
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3547849, 0.3545395
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1195665, 0.1195550
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7315424, 0.7308998
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9558123, 0.9554614
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0721033, 1.0717362

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 2072

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3551

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0282690, upper bound: 0.0282524
time: 103.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282754, upper bound: 0.0282529
time: 10.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7406731, 0.7407297
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5032779, 1.5031562
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6988617, 0.6989016
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2823767, 0.2823777
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7381377, 0.7380766
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3545206, 0.3547541
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1195536, 0.1195713
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7307411, 0.7315000
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9556025, 0.9548668
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0722120, 1.0716347

Time for backsubstitution: 5.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 3224

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282333, upper bound: 0.0282827
time: 104.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0282555, upper bound: 0.0282602
time: 138.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7406002, 0.7408026
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5032398, 1.5031943
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6988592, 0.6989041
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2823068, 0.2824476
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7381509, 0.7380635
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3544905, 0.3547842
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1195561, 0.1195688
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7308114, 0.7314296
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9553084, 0.9551610
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0720696, 1.0717770

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2920

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0282553, upper bound: 0.0282618
time: 115.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282553, upper bound: 0.0282857
time: 10.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7389973, 0.7392120
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5014070, 1.5017939
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6989361, 0.6989294
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2815731, 0.2815376
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7383558, 0.7383568
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3537441, 0.3536869
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1197046, 0.1197048
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7331045, 0.7330964
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9555311, 0.9559946
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0702400, 1.0706915

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 3597
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2354

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282583, upper bound: 0.0282985
time: 150.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282583, upper bound: 0.0282975
time: 141.73 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 298.45 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 298.45
Output dim: 6, lower bound: -0.0282781, upper bound: 0.0282807
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 298.45
Output dim: 6, lower bound: -0.0282798, upper bound: 0.0282769
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 298.45
Output dim: 6, lower bound: -0.0282844, upper bound: 0.0282821
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 298.45
Output dim: 6, lower bound: -0.0282844, upper bound: 0.0282875
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 298.45
Output dim: 6, lower bound: -0.0282828, upper bound: 0.0282827
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 298.45
Output dim: 6, lower bound: -0.0282752, upper bound: 0.0282870
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 298.45
Output dim: 6, lower bound: -0.0282772, upper bound: 0.0282726
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 298.45
Output dim: 6, lower bound: -0.0282683, upper bound: 0.0282822
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 298.45
Output dim: 6, lower bound: -0.0282690, upper bound: 0.0282524
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 298.45
Output dim: 6, lower bound: -0.0282754, upper bound: 0.0282529
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 298.45
Output dim: 6, lower bound: -0.0282333, upper bound: 0.0282827
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 298.45
Output dim: 6, lower bound: -0.0282555, upper bound: 0.0282602
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 298.45
Output dim: 6, lower bound: -0.0282553, upper bound: 0.0282618
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 298.45
Output dim: 6, lower bound: -0.0282553, upper bound: 0.0282857
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 298.45
Output dim: 6, lower bound: -0.0282583, upper bound: 0.0282985
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 298.45
Output dim: 6, lower bound: -0.0282583, upper bound: 0.0282975
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 298.45
Output dim: 6, lower bound: -0.0282573, upper bound: 0.0282996
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 298.45
Output dim: 6, lower bound: -0.0282825, upper bound: 0.0282702
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 298.45
Output dim: 6, lower bound: -0.0282831, upper bound: 0.0282717
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 298.45
Output dim: 6, lower bound: -0.0282860, upper bound: 0.0282866
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 298.45
Output dim: 6, lower bound: -0.0282855, upper bound: 0.0282861
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 298.45
Output dim: 6, lower bound: -0.0282710, upper bound: 0.0282884
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 298.45
Output dim: 6, lower bound: -0.0282860, upper bound: 0.0282751

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 128.77 + 3609.21 = 3737.98 seconds
