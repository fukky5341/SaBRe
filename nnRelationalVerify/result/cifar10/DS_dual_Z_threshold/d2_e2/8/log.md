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
execution time: IAR + RelationalAnalysis = 8.31 + 120.90 = 129.21 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0282980, upper bound: 0.0283015

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 367

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282975, upper bound: 0.0281079
time: 94.25 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0281042, upper bound: 0.0283004
time: 421.93 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 516.27 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 516.27
Output dim: 6, lower bound: -0.0282975, upper bound: 0.0281079
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 516.27
Output dim: 6, lower bound: -0.0281042, upper bound: 0.0283004

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7431341, 0.7429865
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5061700, 1.5063334
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.7005389, 0.7003824
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2770591, 0.2767565
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7393144, 0.7392939
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3474096, 0.3470035
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1176648, 0.1175717
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7286493, 0.7288955
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9609780, 0.9609461
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0697200, 1.0700537

Time for backsubstitution: 5.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 307

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0276764, upper bound: 0.0281085
time: 24.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282951, upper bound: 0.0274896
time: 10.59 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7429864, 0.7431341
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5063334, 1.5061703
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.7003824, 0.7005389
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2767565, 0.2770590
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7392939, 0.7393144
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3470035, 0.3474096
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1175717, 0.1176648
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7288955, 0.7286493
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9609459, 0.9609780
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0700538, 1.0697199

Time for backsubstitution: 5.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 307
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 307

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0274840, upper bound: 0.0276801
time: 133.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0281027, upper bound: 0.0276841
time: 31.84 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 170.92 seconds
DS_DSZ1_DSZ1, status: Status.VERIFIED, split count: 2, time: 170.92
Output dim: 6, lower bound: -0.0276764, upper bound: 0.0281085
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 170.92
Output dim: 6, lower bound: -0.0282951, upper bound: 0.0274896
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 170.92
Output dim: 6, lower bound: -0.0274840, upper bound: 0.0276801
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 170.92
Output dim: 6, lower bound: -0.0281027, upper bound: 0.0276841

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7433957, 0.7432665
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5060606, 1.5062294
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6978031, 0.6977844
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2755002, 0.2752827
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7373775, 0.7372534
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3362923, 0.3364478
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1123127, 0.1118714
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7068936, 0.7082368
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9604313, 0.9604023
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0684047, 1.0688566

Time for backsubstitution: 6.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3515
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3515

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0282542, upper bound: 0.0274883
time: 11.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282939, upper bound: 0.0274476
time: 7.53 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 25.31 seconds
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 25.31
Output dim: 6, lower bound: -0.0282542, upper bound: 0.0274883
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 25.31
Output dim: 6, lower bound: -0.0282939, upper bound: 0.0274476

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7436962, 0.7435798
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5060023, 1.5061741
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6963545, 0.6963897
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2752281, 0.2749961
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7367848, 0.7366845
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3362255, 0.3363775
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1121641, 0.1117151
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7060887, 0.7074692
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9604537, 0.9604261
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0682385, 1.0686988

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3488
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3488

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0282210, upper bound: 0.0274435
time: 28.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282945, upper bound: 0.0273722
time: 13.39 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 47.96 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 47.96
Output dim: 6, lower bound: -0.0282210, upper bound: 0.0274435
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 47.96
Output dim: 6, lower bound: -0.0282945, upper bound: 0.0273722

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7435956, 0.7434750
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5059652, 1.5061340
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6960044, 0.6960613
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2751017, 0.2748757
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7367667, 0.7366678
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3360742, 0.3362275
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1121514, 0.1117026
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7060530, 0.7074298
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9603448, 0.9603102
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0682192, 1.0686750

Time for backsubstitution: 6.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3457

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282877, upper bound: 0.0273679
time: 79.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282915, upper bound: 0.0273608
time: 83.00 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 168.56 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 168.56
Output dim: 6, lower bound: -0.0282877, upper bound: 0.0273679
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 168.56
Output dim: 6, lower bound: -0.0282915, upper bound: 0.0273608

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7435080, 0.7433074
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5059577, 1.5061250
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6960014, 0.6959674
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2750888, 0.2748345
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7367626, 0.7366595
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3360593, 0.3361905
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1121422, 0.1116913
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7059588, 0.7073193
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9603264, 0.9603162
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0681968, 1.0686674

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2376

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282824, upper bound: 0.0273652
time: 10.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282852, upper bound: 0.0273642
time: 127.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7434280, 0.7433873
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5059565, 1.5061269
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6959106, 0.6960570
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2750604, 0.2748621
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7367583, 0.7366636
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3360372, 0.3362075
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1121379, 0.1116934
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7059427, 0.7073253
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9603505, 0.9602919
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0682106, 1.0686526

Time for backsubstitution: 6.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2376

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282831, upper bound: 0.0273590
time: 15.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282868, upper bound: 0.0273611
time: 10.79 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 33.13 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 33.13
Output dim: 6, lower bound: -0.0282824, upper bound: 0.0273652
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 33.13
Output dim: 6, lower bound: -0.0282852, upper bound: 0.0273642
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 33.13
Output dim: 6, lower bound: -0.0282831, upper bound: 0.0273590
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 33.13
Output dim: 6, lower bound: -0.0282868, upper bound: 0.0273611

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7429323, 0.7427121
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5055319, 1.5056918
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6959840, 0.6959499
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2750381, 0.2747841
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7365065, 0.7363961
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3360050, 0.3361364
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1121414, 0.1116905
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7059560, 0.7073163
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9599217, 0.9599018
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0679538, 1.0684221

Time for backsubstitution: 6.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3504

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0282606, upper bound: 0.0273556
time: 62.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0282722, upper bound: 0.0273408
time: 56.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7429128, 0.7427322
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5055243, 1.5056999
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6959838, 0.6959503
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2750385, 0.2747837
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7365044, 0.7364035
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3360052, 0.3361362
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1121414, 0.1116905
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7059558, 0.7073166
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9599119, 0.9599133
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0679514, 1.0684240

Time for backsubstitution: 6.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 3504

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0282645, upper bound: 0.0273534
time: 79.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0282727, upper bound: 0.0273422
time: 61.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7428522, 0.7427921
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5055307, 1.5056932
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6958932, 0.6960394
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2750097, 0.2748117
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7365023, 0.7364002
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3359829, 0.3361533
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1121371, 0.1116926
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7059398, 0.7073224
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9599457, 0.9598775
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0679677, 1.0684073

Time for backsubstitution: 6.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 3504

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0282625, upper bound: 0.0273498
time: 11.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282737, upper bound: 0.0273391
time: 67.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7428327, 0.7428123
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5055231, 1.5057008
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6958929, 0.6960400
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2750101, 0.2748113
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7365004, 0.7364075
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3359830, 0.3361531
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1121371, 0.1116926
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7059396, 0.7073227
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9599360, 0.9598889
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0679655, 1.0684092

Time for backsubstitution: 6.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3504
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3504

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0282657, upper bound: 0.0273517
time: 17.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282776, upper bound: 0.0273380
time: 130.27 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 154.77 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 154.77
Output dim: 6, lower bound: -0.0282606, upper bound: 0.0273556
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 154.77
Output dim: 6, lower bound: -0.0282722, upper bound: 0.0273408
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 154.77
Output dim: 6, lower bound: -0.0282645, upper bound: 0.0273534
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 154.77
Output dim: 6, lower bound: -0.0282727, upper bound: 0.0273422
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 154.77
Output dim: 6, lower bound: -0.0282625, upper bound: 0.0273498
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 154.77
Output dim: 6, lower bound: -0.0282737, upper bound: 0.0273391
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 154.77
Output dim: 6, lower bound: -0.0282657, upper bound: 0.0273517
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 154.77
Output dim: 6, lower bound: -0.0282776, upper bound: 0.0273380

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7428522, 0.7427868
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5055307, 1.5056925
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6958932, 0.6960390
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2750097, 0.2748108
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7364992, 0.7364002
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3359824, 0.3361533
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1121371, 0.1116919
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7059361, 0.7073224
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9599457, 0.9598687
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0679677, 1.0684062

Time for backsubstitution: 6.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2601

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0282674, upper bound: 0.0273308
time: 145.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0282641, upper bound: 0.0273347
time: 13.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.9831285, -2.6686647, -3.9831285, -2.6686647, -0.7428327, 0.7428070
1: -4.4319663, -2.4225478, -4.4319663, -2.4225478, -1.5055231, 1.5057001
2: -0.7337865, 0.0352446, -0.7337865, 0.0352446, -0.6958929, 0.6960396
3: -1.1765749, -0.6218884, -1.1765749, -0.6218884, -0.2750101, 0.2748104
4: -0.7953696, 0.1325753, -0.7953696, 0.1325753, -0.7364972, 0.7364075
5: -1.4758651, -0.6934081, -1.4758651, -0.6934081, -0.3359826, 0.3361531
6: 0.4467425, 0.6884048, 0.4467425, 0.6884048, -0.1121371, 0.1116919
7: -2.4962945, -1.0559874, -2.4962945, -1.0559874, -0.7059358, 0.7073227
8: -4.3948126, -2.7387223, -4.3948126, -2.7387223, -0.9599360, 0.9598799
9: -4.2514644, -2.6344573, -4.2514644, -2.6344573, -1.0679655, 1.0684086

Time for backsubstitution: 6.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 579
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3065
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 124
type: DSZ, layer: 1, pos: 125
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2570
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2436
type: DSZ, layer: 1, pos: 3094
type: DSZ, layer: 1, pos: 2453
type: DSZ, layer: 1, pos: 3092
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2455
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 3249
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3099
type: DSZ, layer: 1, pos: 3519
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 3098
type: DSZ, layer: 1, pos: 2406
type: DSZ, layer: 1, pos: 2452
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3007
type: DSZ, layer: 1, pos: 3499
type: DSZ, layer: 1, pos: 201
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 369
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2408
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2410
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3005
type: DSZ, layer: 1, pos: 3244
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 842
type: DSZ, layer: 1, pos: 368
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2538
type: DSZ, layer: 1, pos: 2583
type: DSZ, layer: 1, pos: 846
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 75
type: DSZ, layer: 1, pos: 2535
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2525
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 228
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 320
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2793
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2943
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 594
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 2929
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3302
type: DSZ, layer: 1, pos: 3260
type: DSZ, layer: 1, pos: 3379
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2318
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 735
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3155
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 3153
type: DSZ, layer: 1, pos: 2492
type: DSZ, layer: 1, pos: 2477
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2252
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 383
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 3083
type: DSZ, layer: 1, pos: 2495
type: DSZ, layer: 1, pos: 3579
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 677
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 309
type: DSZ, layer: 1, pos: 2930
type: DSZ, layer: 1, pos: 3355
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 252
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 3511
type: DSZ, layer: 1, pos: 308
type: DSZ, layer: 1, pos: 3201
type: DSZ, layer: 1, pos: 3202
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3162
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 14
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 180
type: DSZ, layer: 1, pos: 209
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 221
type: DSZ, layer: 1, pos: 222
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 313
type: DSZ, layer: 1, pos: 341
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 448
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 580
type: DSZ, layer: 1, pos: 671
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 713
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 841
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 896
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 1019
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2111
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2187
type: DSZ, layer: 1, pos: 2188
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2260
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2368
type: DSZ, layer: 1, pos: 2416
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2485
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2506
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2626
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2654
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2698
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2919
type: DSZ, layer: 1, pos: 2920
type: DSZ, layer: 1, pos: 2939
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2953
type: DSZ, layer: 1, pos: 2956
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2982
type: DSZ, layer: 1, pos: 3013
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3143
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3148
type: DSZ, layer: 1, pos: 3203
type: DSZ, layer: 1, pos: 3224
type: DSZ, layer: 1, pos: 3327
type: DSZ, layer: 1, pos: 3342
type: DSZ, layer: 1, pos: 3357
type: DSZ, layer: 1, pos: 3358
type: DSZ, layer: 1, pos: 3359
type: DSZ, layer: 1, pos: 3361
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3596
type: DSZ, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2601

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0282712, upper bound: 0.0273323
time: 19.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0282703, upper bound: 0.0273328
time: 11.34 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 37.14 seconds
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 37.14
Output dim: 6, lower bound: -0.0282674, upper bound: 0.0273308
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 37.14
Output dim: 6, lower bound: -0.0282641, upper bound: 0.0273347
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 37.14
Output dim: 6, lower bound: -0.0282712, upper bound: 0.0273323
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 37.14
Output dim: 6, lower bound: -0.0282703, upper bound: 0.0273328

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 129.21 + 1866.15 = 1995.35 seconds
