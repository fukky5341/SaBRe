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
execution time: IAR + RelationalAnalysis = 7.50 + 123.08 = 130.58 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0282980, upper bound: 0.0283015

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 228
type: A, layer: 1, pos: 308
type: A, layer: 1, pos: 307
type: A, layer: 1, pos: 309
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 378
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 3488
type: A, layer: 1, pos: 369
type: A, layer: 1, pos: 368
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3499
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 3515
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3203
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 3202
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 341
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3094
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 383
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 3355
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 3342
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 355
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2982
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 3428
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2257
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 3579
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 3083
type: A, layer: 1, pos: 3155
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2455
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 3245
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 2520
type: A, layer: 1, pos: 2570
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2416
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2943
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2930
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2477
type: A, layer: 1, pos: 3260
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2953
type: A, layer: 1, pos: 252
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 3000
type: A, layer: 1, pos: 3162
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 3358
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2501
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 2793
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 448
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 2919
type: A, layer: 1, pos: 2920
type: A, layer: 1, pos: 2939
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3147
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3224
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3596
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 228

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0278112, upper bound: 0.0283006
time: 10.98 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282948, upper bound: 0.0282981
time: 120.50 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 131.54 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 131.54
Output dim: 6, lower bound: -0.0278112, upper bound: 0.0283006
NS_A2, status: Status.UNKNOWN, split count: 1, time: 131.54
Output dim: 6, lower bound: -0.0282948, upper bound: 0.0282981

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3.9815829, -2.6706860, -3.9829786, -2.6703219, -0.7394667, 0.7405055
1: -4.4289861, -2.4262757, -4.4316468, -2.4257908, -1.5001807, 1.5018377
2: -0.7204639, 0.0265248, -0.7235432, 0.0352443, -0.6856244, 0.6797310
3: -1.1748378, -0.6230406, -1.1752744, -0.6219437, -0.2807845, 0.2800750
4: -0.7888446, 0.1286137, -0.7901460, 0.1325740, -0.7318728, 0.7291802
5: -1.4699972, -0.6973034, -1.4711245, -0.6934407, -0.3490383, 0.3463592
6: 0.4524806, 0.6846436, 0.4513953, 0.6883826, -0.1139539, 0.1112645
7: -2.4892149, -1.0677536, -2.4962265, -1.0655890, -0.7166185, 0.7214742
8: -4.3853931, -2.7529097, -4.3947611, -2.7501383, -0.9392720, 0.9459984
9: -4.2365952, -2.6571257, -4.2514048, -2.6529231, -1.0423714, 1.0529537

Time for backsubstitution: 6.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 308
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 309
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 228
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 3499
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 3515
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 3202
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 341
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3094
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 383
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 3355
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 3342
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 3579
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 3155
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 2520
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2416
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2943
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2477
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2953
type: B, layer: 1, pos: 252
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 3000
type: B, layer: 1, pos: 3162
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 3358
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2501
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 2793
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 448
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2939
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3147
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3224
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3596
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 308

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0278071, upper bound: 0.0276397
time: 7.28 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0278064, upper bound: 0.0282951
time: 63.13 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -3.9831257, -2.6686716, -3.9831262, -2.6686702, -0.7424948, 0.7400010
1: -4.4319649, -2.4226499, -4.4319649, -2.4226432, -1.5034325, 1.5042357
2: -0.7337378, 0.0352445, -0.7337464, 0.0352446, -0.6855333, 0.6989599
3: -1.1765699, -0.6218889, -1.1765709, -0.6218890, -0.2809262, 0.2825756
4: -0.7953577, 0.1325753, -0.7953593, 0.1325753, -0.7307693, 0.7384440
5: -1.4758584, -0.6934084, -1.4758594, -0.6934084, -0.3460119, 0.3549642
6: 0.4467636, 0.6884046, 0.4467596, 0.6884046, -0.1108147, 0.1197024
7: -2.4962904, -1.0560780, -2.4962912, -1.0560615, -0.7333146, 0.7170838
8: -4.3948121, -2.7387776, -4.3948126, -2.7387681, -0.9601526, 0.9398305
9: -4.2514458, -2.6345243, -4.2514482, -2.6345122, -1.0754700, 1.0403833

Time for backsubstitution: 6.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 308
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 309
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 228
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 3499
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 3515
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 3202
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 341
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3094
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 383
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 3355
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 3342
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 3579
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 3155
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 2520
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2416
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2943
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2477
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2953
type: B, layer: 1, pos: 252
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 3000
type: B, layer: 1, pos: 3162
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 3358
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2501
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 2793
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 448
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2939
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3147
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3224
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3596
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 308

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282922, upper bound: 0.0276362
time: 9.85 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282908, upper bound: 0.0282933
time: 126.64 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 142.90 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 142.90
Output dim: 6, lower bound: -0.0278071, upper bound: 0.0276397
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 142.90
Output dim: 6, lower bound: -0.0278064, upper bound: 0.0282951
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 142.90
Output dim: 6, lower bound: -0.0282922, upper bound: 0.0276362
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 142.90
Output dim: 6, lower bound: -0.0282908, upper bound: 0.0282933

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -3.9806981, -2.6706984, -3.9838383, -2.6668115, -0.7438743, 0.7423322
1: -4.4281597, -2.4263046, -4.4306402, -2.4255338, -1.4998305, 1.5013208
2: -0.7204445, 0.0263457, -0.7293632, 0.0353961, -0.6851422, 0.6855870
3: -1.1748270, -0.6232395, -1.1780865, -0.6218153, -0.2809254, 0.2828968
4: -0.7888217, 0.1286136, -0.7916173, 0.1396049, -0.7387787, 0.7301252
5: -1.4699903, -0.6973873, -1.4871340, -0.6933361, -0.3467703, 0.3624237
6: 0.4525096, 0.6846434, 0.4512359, 0.6963370, -0.1218105, 0.1101011
7: -2.4891953, -1.0678859, -2.5303903, -1.0655296, -0.7102234, 0.7554311
8: -4.3850517, -2.7529135, -4.3946772, -2.7478452, -0.9418995, 0.9458849
9: -4.2365088, -2.6575813, -4.2562170, -2.6531188, -1.0419947, 1.0581046

Time for backsubstitution: 6.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 307
type: A, layer: 1, pos: 309
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 378
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 3488
type: A, layer: 1, pos: 308
type: A, layer: 1, pos: 369
type: A, layer: 1, pos: 368
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3499
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 3515
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3203
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 3202
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 341
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3094
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 383
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 3355
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 3342
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 355
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2982
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 3428
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2257
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 3579
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 3083
type: A, layer: 1, pos: 3155
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2455
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 3245
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 2520
type: A, layer: 1, pos: 2570
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2416
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2943
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2930
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2477
type: A, layer: 1, pos: 3260
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2953
type: A, layer: 1, pos: 252
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 3000
type: A, layer: 1, pos: 3162
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 3358
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2501
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 2793
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 448
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 2919
type: A, layer: 1, pos: 2920
type: A, layer: 1, pos: 2939
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3147
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3224
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3596
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 307

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0271853, upper bound: 0.0282953
time: 55.62 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0278066, upper bound: 0.0282971
time: 7.29 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -3.9824755, -2.6686773, -3.9822178, -2.6686780, -0.7417405, 0.7389745
1: -4.4318390, -2.4226809, -4.4317894, -2.4226847, -1.5032568, 1.5040207
2: -0.7336629, 0.0341853, -0.7336426, 0.0337621, -0.6839697, 0.6977897
3: -1.1765631, -0.6224006, -1.1765615, -0.6226048, -0.2801951, 0.2820220
4: -0.7940269, 0.1325752, -0.7934980, 0.1325753, -0.7295039, 0.7367207
5: -1.4758471, -0.6964421, -1.4758439, -0.6976544, -0.3417865, 0.3519379
6: 0.4482516, 0.6883916, 0.4488418, 0.6883865, -0.1092894, 0.1176391
7: -2.4962397, -1.0630395, -2.4962208, -1.0658026, -0.7235067, 0.7100519
8: -4.3943105, -2.7387891, -4.3941116, -2.7387838, -0.9595745, 0.9390382
9: -4.2513247, -2.6353517, -4.2512798, -2.6356702, -1.0741448, 1.0393583

Time for backsubstitution: 6.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 307
type: A, layer: 1, pos: 309
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 378
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 3488
type: A, layer: 1, pos: 308
type: A, layer: 1, pos: 369
type: A, layer: 1, pos: 368
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3499
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 3515
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3203
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 3202
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 341
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3094
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 383
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 3355
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 3342
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 355
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2982
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 3428
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2257
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 3579
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 3083
type: A, layer: 1, pos: 3155
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2455
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 3245
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 2520
type: A, layer: 1, pos: 2570
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2416
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2943
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2930
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2477
type: A, layer: 1, pos: 3260
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2953
type: A, layer: 1, pos: 252
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 3000
type: A, layer: 1, pos: 3162
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 3358
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2501
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 2793
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 448
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 2919
type: A, layer: 1, pos: 2920
type: A, layer: 1, pos: 2939
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3147
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3224
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3596
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 307

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0276701, upper bound: 0.0276043
time: 11.12 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282908, upper bound: 0.0276062
time: 13.93 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -3.9822431, -2.6686840, -3.9839804, -2.6651597, -0.7469001, 0.7418506
1: -4.4311390, -2.4226792, -4.4309592, -2.4223876, -1.5030849, 1.5037189
2: -0.7337186, 0.0350653, -0.7395688, 0.0353964, -0.6850507, 0.7048203
3: -1.1765591, -0.6220881, -1.1793834, -0.6217645, -0.2810682, 0.2853963
4: -0.7953349, 0.1325749, -0.7968335, 0.1396060, -0.7376754, 0.7393884
5: -1.4758514, -0.6934922, -1.4918689, -0.6933048, -0.3437645, 0.3710285
6: 0.4467925, 0.6884045, 0.4465994, 0.6963590, -0.1186710, 0.1185404
7: -2.4962707, -1.0562103, -2.5304539, -1.0559996, -0.7269204, 0.7510410
8: -4.3944693, -2.7387824, -4.3947272, -2.7364738, -0.9627821, 0.9397174
9: -4.2513599, -2.6349800, -4.2562618, -2.6347055, -1.0750941, 1.0455338

Time for backsubstitution: 6.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 307
type: A, layer: 1, pos: 309
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 378
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 3488
type: A, layer: 1, pos: 308
type: A, layer: 1, pos: 369
type: A, layer: 1, pos: 368
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3499
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 3515
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3203
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 3202
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 341
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3094
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 383
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 3355
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 3342
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 355
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2982
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 3428
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2257
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 3579
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 3083
type: A, layer: 1, pos: 3155
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2455
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 3245
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 2520
type: A, layer: 1, pos: 2570
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2416
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2943
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2930
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2477
type: A, layer: 1, pos: 3260
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2953
type: A, layer: 1, pos: 252
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 3000
type: A, layer: 1, pos: 3162
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 3358
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2501
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 2793
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 448
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 2919
type: A, layer: 1, pos: 2920
type: A, layer: 1, pos: 2939
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3147
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3224
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3596
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 307

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0276709, upper bound: 0.0282953
time: 92.68 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282897, upper bound: 0.0282942
time: 75.37 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 174.46 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 174.46
Output dim: 6, lower bound: -0.0271853, upper bound: 0.0282953
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 174.46
Output dim: 6, lower bound: -0.0278066, upper bound: 0.0282971
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 174.46
Output dim: 6, lower bound: -0.0276701, upper bound: 0.0276043
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 174.46
Output dim: 6, lower bound: -0.0282908, upper bound: 0.0276062
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 174.46
Output dim: 6, lower bound: -0.0276709, upper bound: 0.0282953
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 174.46
Output dim: 6, lower bound: -0.0282897, upper bound: 0.0282942

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.9796653, -2.6699543, -3.9832430, -2.6668403, -0.7421917, 0.7407461
1: -4.4279881, -2.4269681, -4.4305992, -2.4260526, -1.4991856, 1.5006342
2: -0.7178141, 0.0204874, -0.7287470, 0.0309185, -0.6783144, 0.6793914
3: -1.1741631, -0.6250094, -1.1779925, -0.6231674, -0.2792925, 0.2812101
4: -0.7850118, 0.1273972, -0.7887826, 0.1396024, -0.7350262, 0.7261258
5: -1.4670018, -0.7062395, -1.4871013, -0.7001937, -0.3370188, 0.3535493
6: 0.4575358, 0.6828716, 0.4551351, 0.6962941, -0.1171935, 0.1050058
7: -2.4833646, -1.0845674, -2.5302525, -1.0784907, -0.6915290, 0.7386724
8: -4.3833752, -2.7534344, -4.3934069, -2.7480190, -0.9398054, 0.9435883
9: -4.2359247, -2.6600435, -4.2561378, -2.6550252, -1.0406512, 1.0561398

Time for backsubstitution: 6.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 309
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 228
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 3499
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 3515
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 3202
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 341
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3094
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 383
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 3355
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 3342
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 3579
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 3155
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 2520
type: B, layer: 1, pos: 2416
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2943
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2477
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2953
type: B, layer: 1, pos: 252
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 3000
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 3162
type: B, layer: 1, pos: 3358
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2501
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 2793
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 448
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2939
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3147
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3224
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3596
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 309

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0271820, upper bound: 0.0278872
time: 18.41 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0271844, upper bound: 0.0282926
time: 132.76 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.9804533, -2.6707082, -3.9836769, -2.6668196, -0.7437451, 0.7425641
1: -4.4281573, -2.4263434, -4.4306388, -2.4255643, -1.4998002, 1.5011783
2: -0.7204365, 0.0263382, -0.7293570, 0.0353909, -0.6851304, 0.6829798
3: -1.1748203, -0.6232406, -1.1780818, -0.6218162, -0.2809085, 0.2813723
4: -0.7888174, 0.1286137, -0.7916136, 0.1396050, -0.7368239, 0.7301196
5: -1.4699863, -0.6973875, -1.4871308, -0.6933360, -0.3467571, 0.3518475
6: 0.4525098, 0.6846435, 0.4512361, 0.6963371, -0.1162500, 0.1100349
7: -2.4891925, -1.0678871, -2.5303879, -1.0655307, -0.7102031, 0.7347209
8: -4.3850231, -2.7529173, -4.3946438, -2.7478478, -0.9413463, 0.9458781
9: -4.2364964, -2.6576395, -4.2562075, -2.6531610, -1.0418687, 1.0566247

Time for backsubstitution: 6.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 309
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 228
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 3499
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 3515
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 3202
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 341
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3094
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 383
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 3355
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 3342
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 3579
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 3155
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 2520
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2416
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2943
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2477
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2953
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 252
type: B, layer: 1, pos: 3000
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 3162
type: B, layer: 1, pos: 3358
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2501
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 2793
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 448
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2939
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3147
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3224
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3596
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 309

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0278043, upper bound: 0.0273661
time: 151.36 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0278036, upper bound: 0.0282935
time: 12.04 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.9822302, -2.6686871, -3.9820333, -2.6686854, -0.7415801, 0.7391663
1: -4.4318366, -2.4227209, -4.4317875, -2.4227152, -1.5032247, 1.5038793
2: -0.7336521, 0.0341763, -0.7336339, 0.0337554, -0.6839572, 0.6951811
3: -1.1765541, -0.6224020, -1.1765547, -0.6226060, -0.2801793, 0.2805521
4: -0.7940215, 0.1325748, -0.7934934, 0.1325752, -0.7275646, 0.7367104
5: -1.4758425, -0.6964422, -1.4758403, -0.6976544, -0.3417727, 0.3413781
6: 0.4482529, 0.6883916, 0.4488428, 0.6883864, -0.1040048, 0.1175793
7: -2.4962366, -1.0630405, -2.4962182, -1.0658059, -0.7234859, 0.6893902
8: -4.3942833, -2.7387934, -4.3940840, -2.7387867, -0.9590222, 0.9390297
9: -4.2513099, -2.6354523, -4.2512679, -2.6357455, -1.0739846, 1.0381488

Time for backsubstitution: 6.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 309
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 228
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 3499
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 3515
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 3202
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 341
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3094
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 383
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 3355
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 3342
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 3579
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 3155
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 2520
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2416
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2943
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2477
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2953
type: B, layer: 1, pos: 252
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 3000
type: B, layer: 1, pos: 3162
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 3358
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2501
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 2793
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 448
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2939
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3147
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3224
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3596
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 309

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282868, upper bound: 0.0273689
time: 112.32 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282871, upper bound: 0.0275951
time: 42.37 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.9812098, -2.6679397, -3.9833865, -2.6651888, -0.7452212, 0.7402664
1: -4.4309645, -2.4233418, -4.4309182, -2.4229054, -1.5024356, 1.5030336
2: -0.7310972, 0.0292071, -0.7389536, 0.0309191, -0.6782272, 0.6986256
3: -1.1758962, -0.6238582, -1.1792897, -0.6231161, -0.2794369, 0.2837107
4: -0.7915250, 0.1313584, -0.7940009, 0.1396033, -0.7339225, 0.7353902
5: -1.4728630, -0.7023447, -1.4918361, -0.7001622, -0.3340144, 0.3621547
6: 0.4518187, 0.6866326, 0.4504984, 0.6963161, -0.1140548, 0.1134467
7: -2.4904404, -1.0728912, -2.5303164, -1.0689604, -0.7082263, 0.7342827
8: -4.3927951, -2.7393014, -4.3934574, -2.7366476, -0.9606887, 0.9374218
9: -4.2507768, -2.6374419, -4.2561817, -2.6366134, -1.0737507, 1.0435698

Time for backsubstitution: 6.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 309
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 228
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 3499
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 3515
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 3202
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 341
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3094
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 383
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 3355
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 3342
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 3579
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 3155
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 2520
type: B, layer: 1, pos: 2416
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2943
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2477
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2953
type: B, layer: 1, pos: 252
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 3000
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 3162
type: B, layer: 1, pos: 3358
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2501
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 2793
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 448
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2939
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3147
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3224
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3596
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 309

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0276649, upper bound: 0.0278893
time: 12.88 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0276696, upper bound: 0.0282944
time: 8.15 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.9819984, -2.6686935, -3.9838192, -2.6651678, -0.7467710, 0.7420825
1: -4.4311366, -2.4227192, -4.4309573, -2.4224176, -1.5030539, 1.5035758
2: -0.7337104, 0.0350578, -0.7395629, 0.0353912, -0.6850391, 0.7022128
3: -1.1765527, -0.6220893, -1.1793785, -0.6217654, -0.2810514, 0.2838720
4: -0.7953312, 0.1325752, -0.7968311, 0.1396059, -0.7357206, 0.7393827
5: -1.4758471, -0.6934921, -1.4918655, -0.6933049, -0.3437512, 0.3604524
6: 0.4467928, 0.6884046, 0.4465995, 0.6963590, -0.1131115, 0.1184738
7: -2.4962678, -1.0562112, -2.5304518, -1.0560006, -0.7269004, 0.7303305
8: -4.3944426, -2.7387853, -4.3946953, -2.7364764, -0.9622288, 0.9397111
9: -4.2513475, -2.6350379, -4.2562518, -2.6347489, -1.0749674, 1.0440543

Time for backsubstitution: 6.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 309
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 228
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 3499
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 3515
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 3202
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 341
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3094
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 383
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 3355
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 3342
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 3579
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 3155
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 2520
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2416
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2943
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2477
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2953
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 252
type: B, layer: 1, pos: 3000
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 3162
type: B, layer: 1, pos: 3358
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2501
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 2793
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 448
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2939
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3147
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3224
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3596
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 309

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282871, upper bound: 0.0278897
time: 17.60 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282874, upper bound: 0.0282921
time: 90.92 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 115.20 seconds
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 115.20
Output dim: 6, lower bound: -0.0271820, upper bound: 0.0278872
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 115.20
Output dim: 6, lower bound: -0.0271844, upper bound: 0.0282926
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 115.20
Output dim: 6, lower bound: -0.0278043, upper bound: 0.0273661
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 115.20
Output dim: 6, lower bound: -0.0278036, upper bound: 0.0282935
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 115.20
Output dim: 6, lower bound: -0.0282868, upper bound: 0.0273689
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 115.20
Output dim: 6, lower bound: -0.0282871, upper bound: 0.0275951
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 115.20
Output dim: 6, lower bound: -0.0276649, upper bound: 0.0278893
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 115.20
Output dim: 6, lower bound: -0.0276696, upper bound: 0.0282944
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 115.20
Output dim: 6, lower bound: -0.0282871, upper bound: 0.0278897
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 115.20
Output dim: 6, lower bound: -0.0282874, upper bound: 0.0282921

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.9794507, -2.6699562, -3.9829662, -2.6668432, -0.7421790, 0.7367726
1: -4.4279795, -2.4270260, -4.4305892, -2.4261289, -1.4990944, 1.5006020
2: -0.7174982, 0.0204873, -0.7282574, 0.0309185, -0.6779677, 0.6814511
3: -1.1741606, -0.6250272, -1.1779897, -0.6231869, -0.2792311, 0.2812080
4: -0.7850050, 0.1273969, -0.7887746, 0.1396021, -0.7349886, 0.7234013
5: -1.4670000, -0.7062487, -1.4870992, -0.7002054, -0.3298390, 0.3535450
6: 0.4575423, 0.6828706, 0.4551435, 0.6962928, -0.1170862, 0.1020391
7: -2.4833591, -1.0846379, -2.5302472, -1.0785792, -0.6728877, 0.7382537
8: -4.3833723, -2.7534597, -4.3934031, -2.7480571, -0.9402984, 0.9434507
9: -4.2359123, -2.6600842, -4.2561207, -2.6550794, -1.0373679, 1.0561192

Time for backsubstitution: 6.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 378
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 3488
type: A, layer: 1, pos: 308
type: A, layer: 1, pos: 369
type: A, layer: 1, pos: 368
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3499
type: A, layer: 1, pos: 309
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 3515
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3203
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 3202
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 341
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3094
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 383
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 3355
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 3342
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 355
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2982
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 3428
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2257
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 3579
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 3083
type: A, layer: 1, pos: 3155
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2455
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 3245
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 2520
type: A, layer: 1, pos: 2570
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2416
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2943
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2930
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2477
type: A, layer: 1, pos: 3260
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2953
type: A, layer: 1, pos: 252
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 3000
type: A, layer: 1, pos: 3162
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 3358
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2501
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 2793
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 448
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 2919
type: A, layer: 1, pos: 2920
type: A, layer: 1, pos: 2939
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3147
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3224
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3596
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 237

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0270272, upper bound: 0.0282943
time: 31.02 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0271845, upper bound: 0.0282896
time: 62.88 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.9802394, -2.6707098, -3.9833999, -2.6668222, -0.7437323, 0.7385906
1: -4.4281483, -2.4264026, -4.4306283, -2.4256396, -1.4997084, 1.5011456
2: -0.7201191, 0.0263380, -0.7288657, 0.0353909, -0.6847896, 0.6850356
3: -1.1748180, -0.6232585, -1.1780787, -0.6218359, -0.2808469, 0.2813703
4: -0.7888107, 0.1286137, -0.7916063, 0.1396048, -0.7367864, 0.7273934
5: -1.4699845, -0.6973965, -1.4871290, -0.6933477, -0.3395762, 0.3518433
6: 0.4525162, 0.6846424, 0.4512444, 0.6963358, -0.1161427, 0.1070676
7: -2.4891872, -1.0679576, -2.5303831, -1.0656188, -0.6915616, 0.7343025
8: -4.3850203, -2.7529423, -4.3946400, -2.7478864, -0.9418376, 0.9457409
9: -4.2364826, -2.6576805, -4.2561908, -2.6532145, -1.0385852, 1.0566032

Time for backsubstitution: 6.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 378
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 3488
type: A, layer: 1, pos: 308
type: A, layer: 1, pos: 369
type: A, layer: 1, pos: 368
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3499
type: A, layer: 1, pos: 309
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 3515
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3203
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 3202
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 341
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3094
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 383
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 3355
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 3342
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 355
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2982
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 3428
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2257
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 3579
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 3083
type: A, layer: 1, pos: 3155
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2455
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 3245
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 2520
type: A, layer: 1, pos: 2570
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2416
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2943
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2930
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2477
type: A, layer: 1, pos: 3260
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2953
type: A, layer: 1, pos: 252
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 3000
type: A, layer: 1, pos: 3162
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 3358
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2501
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 2793
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 448
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 2919
type: A, layer: 1, pos: 2920
type: A, layer: 1, pos: 2939
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3147
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3224
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3596
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 237

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0276460, upper bound: 0.0282924
time: 12.22 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0278026, upper bound: 0.0282955
time: 11.19 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3.9799089, -2.6687160, -3.9789138, -2.6671324, -0.7376936, 0.7337829
1: -4.4316874, -2.4234724, -4.4315281, -2.4236655, -1.5021210, 1.5028231
2: -0.7327732, 0.0341560, -0.7323298, 0.0344816, -0.6839924, 0.6933255
3: -1.1765295, -0.6226565, -1.1766227, -0.6230308, -0.2797257, 0.2804811
4: -0.7907010, 0.1325741, -0.7889317, 0.1309793, -0.7237338, 0.7326665
5: -1.4758210, -0.7016812, -1.4734385, -0.7043836, -0.3353071, 0.3348384
6: 0.4513209, 0.6882912, 0.4527560, 0.6869168, -0.1004220, 0.1142906
7: -2.4959168, -1.0765952, -2.4899802, -1.0831113, -0.7063740, 0.6699451
8: -4.3941803, -2.7390978, -4.3939543, -2.7392690, -0.9581919, 0.9384733
9: -4.2509985, -2.6387808, -4.2495847, -2.6400719, -1.0696483, 1.0339622

Time for backsubstitution: 6.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 378
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 3488
type: A, layer: 1, pos: 308
type: A, layer: 1, pos: 369
type: A, layer: 1, pos: 368
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3499
type: A, layer: 1, pos: 309
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 3515
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3203
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 3202
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 341
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3094
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 383
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 3355
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 3342
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 355
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2982
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 3428
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2257
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 3579
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 3083
type: A, layer: 1, pos: 3155
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2455
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 3245
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 2520
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2570
type: A, layer: 1, pos: 2416
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2943
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2930
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2477
type: A, layer: 1, pos: 3260
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2953
type: A, layer: 1, pos: 252
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 3000
type: A, layer: 1, pos: 3162
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 3358
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2501
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 2793
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 448
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 2919
type: A, layer: 1, pos: 2920
type: A, layer: 1, pos: 2939
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3147
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3224
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3596
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 237

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0281264, upper bound: 0.0273712
time: 9.97 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282846, upper bound: 0.0273708
time: 10.02 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3.9819732, -2.6686893, -3.9816892, -2.6686881, -0.7415669, 0.7351408
1: -4.4318285, -2.4227800, -4.4317760, -2.4227924, -1.5031295, 1.5038455
2: -0.7333339, 0.0341763, -0.7332147, 0.0337552, -0.6835519, 0.6968831
3: -1.1765521, -0.6224309, -1.1765518, -0.6226441, -0.2800993, 0.2805310
4: -0.7940139, 0.1325752, -0.7934825, 0.1325752, -0.7275258, 0.7339584
5: -1.4758403, -0.6964518, -1.4758372, -0.6976671, -0.3346323, 0.3413732
6: 0.4482604, 0.6883904, 0.4488524, 0.6883852, -0.1038894, 0.1145272
7: -2.4962296, -1.0631118, -2.4962082, -1.0658944, -0.7052177, 0.6889674
8: -4.3942785, -2.7388184, -4.3940792, -2.7388182, -0.9593087, 0.9388618
9: -4.2512932, -2.6354933, -4.2512465, -2.6357994, -1.0707389, 1.0381191

Time for backsubstitution: 6.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 378
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 3488
type: A, layer: 1, pos: 308
type: A, layer: 1, pos: 369
type: A, layer: 1, pos: 368
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3499
type: A, layer: 1, pos: 309
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 3515
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3203
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 3202
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 341
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3094
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 383
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 3355
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 3342
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 355
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2982
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 3428
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2257
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 3579
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 3083
type: A, layer: 1, pos: 3155
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2455
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 3245
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 2520
type: A, layer: 1, pos: 2570
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2416
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2943
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2930
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2477
type: A, layer: 1, pos: 3260
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2953
type: A, layer: 1, pos: 252
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 3000
type: A, layer: 1, pos: 3162
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 3358
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2501
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 2793
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 448
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 2919
type: A, layer: 1, pos: 2920
type: A, layer: 1, pos: 2939
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3147
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3224
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3596
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 237

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0281316, upper bound: 0.0275984
time: 9.58 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282879, upper bound: 0.0275944
time: 30.01 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.9809957, -2.6679420, -3.9831104, -2.6651912, -0.7452085, 0.7362951
1: -4.4309549, -2.4234023, -4.4309068, -2.4229817, -1.5023435, 1.5030017
2: -0.7307813, 0.0292071, -0.7384660, 0.0309190, -0.6778785, 0.7006894
3: -1.1758940, -0.6238758, -1.1792871, -0.6231365, -0.2793773, 0.2837086
4: -0.7915181, 0.1313584, -0.7939934, 0.1396033, -0.7338854, 0.7326686
5: -1.4728614, -0.7023537, -1.4918340, -0.7001741, -0.3268403, 0.3621505
6: 0.4518250, 0.6866316, 0.4505066, 0.6963148, -0.1139475, 0.1104839
7: -2.4904346, -1.0729618, -2.5303116, -1.0690489, -0.6895769, 0.7338635
8: -4.3927913, -2.7393258, -4.3934526, -2.7366858, -0.9611819, 0.9372851
9: -4.2507634, -2.6374829, -4.2561646, -2.6366661, -1.0704679, 1.0435486

Time for backsubstitution: 6.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 378
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 3488
type: A, layer: 1, pos: 308
type: A, layer: 1, pos: 369
type: A, layer: 1, pos: 368
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3499
type: A, layer: 1, pos: 309
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 3515
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3203
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 3202
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 341
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3094
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 383
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 3355
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 3342
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 355
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2982
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 3428
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2257
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 3579
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 3083
type: A, layer: 1, pos: 3155
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2455
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 3245
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 2520
type: A, layer: 1, pos: 2570
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2416
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2943
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2930
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2477
type: A, layer: 1, pos: 3260
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2953
type: A, layer: 1, pos: 252
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 3000
type: A, layer: 1, pos: 3162
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 3358
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2501
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 2793
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 448
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 2919
type: A, layer: 1, pos: 2920
type: A, layer: 1, pos: 2939
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3147
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3224
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3596
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 237

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0275110, upper bound: 0.0282957
time: 8.09 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0276687, upper bound: 0.0282923
time: 7.65 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3.9791803, -2.6687231, -3.9801307, -2.6645086, -0.7421631, 0.7365757
1: -4.4309874, -2.4234774, -4.4306960, -2.4233840, -1.5019302, 1.5025139
2: -0.7328256, 0.0350373, -0.7384522, 0.0363713, -0.6852733, 0.7003344
3: -1.1765270, -0.6223770, -1.1793519, -0.6222242, -0.2806197, 0.2835032
4: -0.7920035, 0.1325738, -0.7922914, 0.1380075, -0.7318771, 0.7353784
5: -1.4758244, -0.6987335, -1.4894524, -0.7000320, -0.3372532, 0.3531884
6: 0.4498645, 0.6883041, 0.4505515, 0.6948408, -0.1089386, 0.1152560
7: -2.4959402, -1.0697701, -2.5241847, -1.0733172, -0.7094437, 0.7098371
8: -4.3943396, -2.7390921, -4.3945909, -2.7369592, -0.9613032, 0.9392357
9: -4.2510080, -2.6383677, -4.2545176, -2.6390672, -1.0705843, 1.0394876

Time for backsubstitution: 6.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 378
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 3488
type: A, layer: 1, pos: 308
type: A, layer: 1, pos: 369
type: A, layer: 1, pos: 368
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3499
type: A, layer: 1, pos: 309
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 3515
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3203
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 3202
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 341
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3094
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 383
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 3355
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 3342
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 355
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2982
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 3428
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2257
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 3579
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 3083
type: A, layer: 1, pos: 3155
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2455
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 3245
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 2520
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2570
type: A, layer: 1, pos: 2416
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2943
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2930
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2477
type: A, layer: 1, pos: 3260
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2953
type: A, layer: 1, pos: 252
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 3000
type: A, layer: 1, pos: 3162
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 3358
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2501
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 2793
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 448
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 2919
type: A, layer: 1, pos: 2920
type: A, layer: 1, pos: 2939
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3147
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3224
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3596
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 237

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0281286, upper bound: 0.0278891
time: 15.74 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282846, upper bound: 0.0278877
time: 105.72 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.9817848, -2.6686957, -3.9835429, -2.6651702, -0.7467584, 0.7381108
1: -4.4311280, -2.4227767, -4.4309464, -2.4224930, -1.5029614, 1.5035439
2: -0.7333935, 0.0350578, -0.7390735, 0.0353910, -0.6846967, 0.7042724
3: -1.1765501, -0.6221071, -1.1793758, -0.6217855, -0.2809914, 0.2838700
4: -0.7953246, 0.1325752, -0.7968231, 0.1396061, -0.7356835, 0.7366594
5: -1.4758456, -0.6935010, -1.4918635, -0.6933166, -0.3365761, 0.3604482
6: 0.4467992, 0.6884035, 0.4466078, 0.6963578, -0.1130041, 0.1155105
7: -2.4962623, -1.0562820, -2.5304470, -1.0560890, -0.7082503, 0.7299117
8: -4.3944387, -2.7388108, -4.3946896, -2.7365150, -0.9627210, 0.9395742
9: -4.2513342, -2.6350784, -4.2562356, -2.6348019, -1.0716847, 1.0440326

Time for backsubstitution: 6.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 378
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 3488
type: A, layer: 1, pos: 308
type: A, layer: 1, pos: 369
type: A, layer: 1, pos: 368
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3499
type: A, layer: 1, pos: 309
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 3515
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3203
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 3202
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 341
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3094
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 383
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 3355
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 3342
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 355
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2982
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 3428
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2257
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 3579
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 3083
type: A, layer: 1, pos: 3155
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2455
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 3245
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 2520
type: A, layer: 1, pos: 2570
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2416
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2943
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2930
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2477
type: A, layer: 1, pos: 3260
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2953
type: A, layer: 1, pos: 252
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 3000
type: A, layer: 1, pos: 3162
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 3358
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2501
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 2793
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 448
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 2919
type: A, layer: 1, pos: 2920
type: A, layer: 1, pos: 2939
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3147
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3224
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3596
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 237

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0281305, upper bound: 0.0282938
time: 11.42 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282865, upper bound: 0.0282922
time: 106.12 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 124.20 seconds
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 124.20
Output dim: 6, lower bound: -0.0270272, upper bound: 0.0282943
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 124.20
Output dim: 6, lower bound: -0.0271845, upper bound: 0.0282896
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 124.20
Output dim: 6, lower bound: -0.0276460, upper bound: 0.0282924
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 124.20
Output dim: 6, lower bound: -0.0278026, upper bound: 0.0282955
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 124.20
Output dim: 6, lower bound: -0.0281264, upper bound: 0.0273712
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 124.20
Output dim: 6, lower bound: -0.0282846, upper bound: 0.0273708
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 124.20
Output dim: 6, lower bound: -0.0281316, upper bound: 0.0275984
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 124.20
Output dim: 6, lower bound: -0.0282879, upper bound: 0.0275944
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 124.20
Output dim: 6, lower bound: -0.0275110, upper bound: 0.0282957
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 124.20
Output dim: 6, lower bound: -0.0276687, upper bound: 0.0282923
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 124.20
Output dim: 6, lower bound: -0.0281286, upper bound: 0.0278891
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 124.20
Output dim: 6, lower bound: -0.0282846, upper bound: 0.0278877
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 124.20
Output dim: 6, lower bound: -0.0281305, upper bound: 0.0282938
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 124.20
Output dim: 6, lower bound: -0.0282865, upper bound: 0.0282922

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.9789009, -2.6702752, -3.9825215, -2.6668441, -0.7415405, 0.7358043
1: -4.4267602, -2.4283919, -4.4305401, -2.4272602, -1.4968625, 1.4992888
2: -0.7142102, 0.0178074, -0.7256247, 0.0309164, -0.6746063, 0.6760161
3: -1.1737976, -0.6253106, -1.1777327, -0.6231903, -0.2788848, 0.2806588
4: -0.7824733, 0.1254002, -0.7867340, 0.1396019, -0.7325362, 0.7195569
5: -1.4654276, -0.7075224, -1.4858236, -0.7002103, -0.3282649, 0.3509844
6: 0.4592434, 0.6814868, 0.4565303, 0.6962861, -0.1153927, 0.0992609
7: -2.4814174, -1.0872645, -2.5302300, -1.0807264, -0.6687264, 0.7356600
8: -4.3810334, -2.7563286, -4.3933907, -2.7503970, -0.9356588, 0.9406105
9: -4.2304392, -2.6668267, -4.2560387, -2.6605854, -1.0266181, 1.0495167

Time for backsubstitution: 6.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 228
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 3499
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 3515
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 3202
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 341
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3094
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 383
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 3355
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 3342
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 3579
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 3155
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 2520
type: B, layer: 1, pos: 2416
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2943
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2477
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2953
type: B, layer: 1, pos: 252
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 3000
type: B, layer: 1, pos: 3162
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 3358
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2501
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 2793
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 448
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2939
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3147
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3224
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3596
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 378

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0268430, upper bound: 0.0275941
time: 68.87 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0270265, upper bound: 0.0282917
time: 239.86 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.9793849, -2.6699564, -3.9829049, -2.6668437, -0.7411894, 0.7367582
1: -4.4279795, -2.4270520, -4.4305887, -2.4261515, -1.4987601, 1.4989543
2: -0.7174616, 0.0204872, -0.7282276, 0.0309184, -0.6733469, 0.6814407
3: -1.1741598, -0.6250277, -1.1779890, -0.6231869, -0.2789561, 0.2812065
4: -0.7849921, 0.1273969, -0.7887644, 0.1396021, -0.7315063, 0.7233734
5: -1.4669962, -0.7062488, -1.4870963, -0.7002053, -0.3272476, 0.3535425
6: 0.4575481, 0.6828687, 0.4551481, 0.6962913, -0.1141909, 0.1020271
7: -2.4833536, -1.0846516, -2.5302429, -1.0785904, -0.6727933, 0.7342330
8: -4.3833723, -2.7534676, -4.3934031, -2.7480648, -0.9402857, 0.9386969
9: -4.2358847, -2.6601048, -4.2560978, -2.6550961, -1.0373313, 1.0455011

Time for backsubstitution: 6.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 228
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 3499
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 3515
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 3202
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 341
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3094
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 383
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 3355
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 3342
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 3579
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 3155
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 2520
type: B, layer: 1, pos: 2416
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2943
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2477
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2953
type: B, layer: 1, pos: 252
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 3000
type: B, layer: 1, pos: 3162
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 3358
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2501
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 2793
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 448
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2939
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3147
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3224
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3596
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 378

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0270009, upper bound: 0.0282918
time: 11.93 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0271823, upper bound: 0.0282915
time: 133.32 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.9796891, -2.6710286, -3.9829555, -2.6668227, -0.7430938, 0.7376223
1: -4.4269285, -2.4277675, -4.4305792, -2.4267719, -1.4974751, 1.4998322
2: -0.7168338, 0.0236582, -0.7262346, 0.0353888, -0.6814293, 0.6796000
3: -1.1744556, -0.6235418, -1.1778219, -0.6218390, -0.2805008, 0.2808210
4: -0.7862784, 0.1266167, -0.7895657, 0.1396044, -0.7343332, 0.7235494
5: -1.4684120, -0.6986701, -1.4858534, -0.6933528, -0.3380023, 0.3492828
6: 0.4542172, 0.6832586, 0.4526311, 0.6963292, -0.1144492, 0.1042894
7: -2.4872451, -1.0705841, -2.5303655, -1.0677664, -0.6873993, 0.7317089
8: -4.3826818, -2.7558110, -4.3946276, -2.7502255, -0.9371980, 0.9429014
9: -4.2310104, -2.6644235, -4.2561083, -2.6587205, -1.0278347, 1.0500019

Time for backsubstitution: 6.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 228
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 3499
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 3515
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 3202
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 341
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3094
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 383
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 3355
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 3342
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 3579
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 3155
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 2520
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2416
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2943
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2477
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2953
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 252
type: B, layer: 1, pos: 3000
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 3162
type: B, layer: 1, pos: 3358
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2501
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 2793
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 448
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2939
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3147
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3224
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3596
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 378

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0274646, upper bound: 0.0278853
time: 124.62 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0276459, upper bound: 0.0282919
time: 17.46 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.9801736, -2.6707103, -3.9833384, -2.6668222, -0.7427428, 0.7385757
1: -4.4281468, -2.4264281, -4.4306273, -2.4256618, -1.4993742, 1.4994984
2: -0.7200826, 0.0263377, -0.7288360, 0.0353906, -0.6801691, 0.6850248
3: -1.1748173, -0.6232588, -1.1780782, -0.6218364, -0.2805719, 0.2813688
4: -0.7887982, 0.1286137, -0.7915953, 0.1396048, -0.7333041, 0.7273655
5: -1.4699806, -0.6973965, -1.4871258, -0.6933479, -0.3369850, 0.3518409
6: 0.4525219, 0.6846406, 0.4512490, 0.6963342, -0.1132473, 0.1070556
7: -2.4891818, -1.0679712, -2.5303788, -1.0656302, -0.6914671, 0.7302812
8: -4.3850198, -2.7529504, -4.3946385, -2.7478940, -0.9418249, 0.9409876
9: -4.2364559, -2.6577005, -4.2561679, -2.6532309, -1.0385478, 1.0459851

Time for backsubstitution: 6.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 228
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 3499
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 3515
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 3202
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 341
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3094
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 383
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 3355
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 3342
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 3579
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 3155
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 2520
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2416
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2943
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2477
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2953
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 252
type: B, layer: 1, pos: 3000
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 3162
type: B, layer: 1, pos: 3358
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2501
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 2793
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 448
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2939
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3147
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3224
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3596
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 378

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0276195, upper bound: 0.0282913
time: 11.84 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0278029, upper bound: 0.0282930
time: 26.37 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.9798431, -2.6687164, -3.9788523, -2.6671329, -0.7367051, 0.7337692
1: -4.4316869, -2.4234984, -4.4315271, -2.4236882, -1.5017869, 1.5011771
2: -0.7327366, 0.0341557, -0.7323000, 0.0344813, -0.6793721, 0.6933153
3: -1.1765285, -0.6226568, -1.1766220, -0.6230311, -0.2794506, 0.2804796
4: -0.7906884, 0.1325741, -0.7889211, 0.1309793, -0.7202517, 0.7326382
5: -1.4758174, -0.7016814, -1.4734355, -0.7043837, -0.3327158, 0.3348359
6: 0.4513267, 0.6882893, 0.4527607, 0.6869153, -0.0975266, 0.1142787
7: -2.4959114, -1.0766084, -2.4899757, -1.0831225, -0.7062798, 0.6659244
8: -4.3941803, -2.7391062, -4.3939543, -2.7392750, -0.9581788, 0.9337194
9: -4.2509723, -2.6388011, -4.2495613, -2.6400888, -1.0696108, 1.0233444

Time for backsubstitution: 6.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 228
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 3499
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 3515
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 3202
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 341
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3094
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 383
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 3355
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 3342
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 3579
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 3155
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 2520
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2416
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2943
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2477
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2953
type: B, layer: 1, pos: 252
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 3000
type: B, layer: 1, pos: 3162
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 3358
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2501
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 2793
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 448
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2939
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3147
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3224
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3596
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 378

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0281025, upper bound: 0.0273698
time: 12.55 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282857, upper bound: 0.0273688
time: 123.55 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.9819067, -2.6686893, -3.9816279, -2.6686885, -0.7405782, 0.7351267
1: -4.4318275, -2.4228063, -4.4317760, -2.4228158, -1.5027959, 1.5021992
2: -0.7332973, 0.0341761, -0.7331849, 0.0337549, -0.6789313, 0.6968726
3: -1.1765513, -0.6224312, -1.1765509, -0.6226441, -0.2798243, 0.2805295
4: -0.7940009, 0.1325752, -0.7934718, 0.1325752, -0.7240440, 0.7339304
5: -1.4758368, -0.6964520, -1.4758341, -0.6976674, -0.3320411, 0.3413707
6: 0.4482661, 0.6883887, 0.4488571, 0.6883836, -0.1009941, 0.1145151
7: -2.4962246, -1.0631251, -2.4962039, -1.0659057, -0.7051235, 0.6849444
8: -4.3942785, -2.7388265, -4.3940792, -2.7388248, -0.9592957, 0.9341083
9: -4.2512660, -2.6355133, -4.2512221, -2.6358159, -1.0707016, 1.0275011

Time for backsubstitution: 6.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 228
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 3499
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 3515
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 3202
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 341
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3094
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 383
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 3355
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 3342
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 3579
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 3155
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 2520
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2416
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2943
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2477
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2953
type: B, layer: 1, pos: 252
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 3000
type: B, layer: 1, pos: 3162
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 3358
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2501
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 2793
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 448
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2939
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3147
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3224
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3596
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 378

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0281051, upper bound: 0.0275912
time: 114.70 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282864, upper bound: 0.0275945
time: 71.55 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.9804466, -2.6682606, -3.9826655, -2.6651919, -0.7445711, 0.7353272
1: -4.4297314, -2.4247675, -4.4308586, -2.4241149, -1.5001076, 1.5016890
2: -0.7274932, 0.0265269, -0.7358315, 0.0309169, -0.6745175, 0.6952534
3: -1.1755317, -0.6241590, -1.1790297, -0.6231396, -0.2790312, 0.2831593
4: -0.7889864, 0.1293613, -0.7919520, 0.1396033, -0.7314330, 0.7288244
5: -1.4712888, -0.7036273, -1.4905586, -0.7001790, -0.3252663, 0.3595900
6: 0.4535261, 0.6852478, 0.4518936, 0.6963083, -0.1122540, 0.1077057
7: -2.4884934, -1.0755895, -2.5302944, -1.0711966, -0.6854147, 0.7312684
8: -4.3904524, -2.7421954, -4.3934412, -2.7390251, -0.9565419, 0.9344449
9: -4.2452903, -2.6442256, -4.2560835, -2.6421723, -1.0597180, 1.0369470

Time for backsubstitution: 6.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 228
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 3499
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 3515
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 3202
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 341
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3094
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 383
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 3355
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 3342
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 3579
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 3155
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 2520
type: B, layer: 1, pos: 2416
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2943
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2477
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2953
type: B, layer: 1, pos: 252
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 3000
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 3162
type: B, layer: 1, pos: 3358
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2501
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 2793
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 448
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2939
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3147
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3224
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3596
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 378

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0273272, upper bound: 0.0282900
time: 47.10 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0275097, upper bound: 0.0282918
time: 12.69 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.9809303, -2.6679420, -3.9830484, -2.6651912, -0.7442198, 0.7362804
1: -4.4309549, -2.4234283, -4.4309072, -2.4230049, -1.5020099, 1.5013547
2: -0.7307446, 0.0292069, -0.7384363, 0.0309187, -0.6732579, 0.7006787
3: -1.1758932, -0.6238755, -1.1792860, -0.6231365, -0.2791023, 0.2837072
4: -0.7915053, 0.1313584, -0.7939826, 0.1396033, -0.7304031, 0.7326404
5: -1.4728577, -0.7023538, -1.4918311, -0.7001740, -0.3242490, 0.3621480
6: 0.4518307, 0.6866297, 0.4505113, 0.6963133, -0.1110522, 0.1104719
7: -2.4904294, -1.0729754, -2.5303073, -1.0690598, -0.6894825, 0.7298415
8: -4.3927908, -2.7393351, -4.3934526, -2.7366934, -0.9611691, 0.9325311
9: -4.2507362, -2.6375031, -4.2561417, -2.6366830, -1.0704315, 1.0329304

Time for backsubstitution: 6.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 228
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 3499
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 3515
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 3202
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 341
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3094
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 383
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 3355
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 3342
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 3579
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 3155
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 2520
type: B, layer: 1, pos: 2416
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2943
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2477
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2953
type: B, layer: 1, pos: 252
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 3000
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 3162
type: B, layer: 1, pos: 3358
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2501
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 2793
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 448
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2939
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3147
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3224
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3596
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 378

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0274868, upper bound: 0.0282935
time: 12.00 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0276683, upper bound: 0.0278853
time: 93.58 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.9791143, -2.6687231, -3.9800692, -2.6645095, -0.7411743, 0.7365607
1: -4.4309859, -2.4235034, -4.4306955, -2.4234071, -1.5015963, 1.5008674
2: -0.7327895, 0.0350371, -0.7384227, 0.0363711, -0.6806526, 0.7003238
3: -1.1765261, -0.6223772, -1.1793512, -0.6222243, -0.2803447, 0.2835018
4: -0.7919909, 0.1325738, -0.7922806, 0.1380075, -0.7283949, 0.7353501
5: -1.4758208, -0.6987338, -1.4894494, -0.7000321, -0.3346618, 0.3531858
6: 0.4498701, 0.6883023, 0.4505562, 0.6948393, -0.1060434, 0.1152440
7: -2.4959347, -1.0697837, -2.5241804, -1.0733283, -0.7093495, 0.7058166
8: -4.3943391, -2.7391000, -4.3945904, -2.7369659, -0.9612902, 0.9344817
9: -4.2509813, -2.6383882, -4.2544942, -2.6390834, -1.0705473, 1.0288692

Time for backsubstitution: 6.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 228
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 3499
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 3515
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 3202
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 341
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3094
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 383
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 3355
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 3342
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 3579
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 3155
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 2520
type: B, layer: 1, pos: 2416
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2943
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2477
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2953
type: B, layer: 1, pos: 252
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 3000
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 3162
type: B, layer: 1, pos: 3358
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2501
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 2793
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 448
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2939
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3147
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3224
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3596
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 378

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0281015, upper bound: 0.0278834
time: 122.77 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282854, upper bound: 0.0278877
time: 15.77 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.9812350, -2.6690140, -3.9830980, -2.6651704, -0.7461212, 0.7371429
1: -4.4299035, -2.4241426, -4.4308987, -2.4236250, -1.5007250, 1.5022309
2: -0.7301077, 0.0323777, -0.7364399, 0.0353888, -0.6813362, 0.6988369
3: -1.1761885, -0.6223906, -1.1791186, -0.6217886, -0.2806454, 0.2833206
4: -0.7927918, 0.1305782, -0.7947824, 0.1396058, -0.7332301, 0.7328154
5: -1.4742732, -0.6947751, -1.4905882, -0.6933217, -0.3350022, 0.3578879
6: 0.4485002, 0.6870197, 0.4479948, 0.6963511, -0.1113106, 0.1127323
7: -2.4943206, -1.0589099, -2.5304298, -1.0582367, -0.7040876, 0.7273166
8: -4.3921003, -2.7416801, -4.3946781, -2.7388542, -0.9580814, 0.9367337
9: -4.2458620, -2.6418209, -4.2561541, -2.6403089, -1.0609343, 1.0374316

Time for backsubstitution: 6.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 228
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 3499
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 3515
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 3202
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 341
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3094
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 383
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 3355
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 3342
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 3579
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 3155
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 2520
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2416
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2943
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2477
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2953
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 252
type: B, layer: 1, pos: 3000
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 3162
type: B, layer: 1, pos: 3358
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2501
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 2793
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 448
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2939
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3147
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3224
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3596
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 378

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0279470, upper bound: 0.0282908
time: 92.98 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0281275, upper bound: 0.0282901
time: 116.39 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.9817185, -2.6686954, -3.9834809, -2.6651704, -0.7457694, 0.7380959
1: -4.4311275, -2.4228034, -4.4309459, -2.4225161, -1.5026277, 1.5018969
2: -0.7333571, 0.0350575, -0.7390438, 0.0353909, -0.6800760, 0.7042620
3: -1.1765493, -0.6221075, -1.1793748, -0.6217854, -0.2807164, 0.2838685
4: -0.7953110, 0.1325752, -0.7968127, 0.1396061, -0.7322012, 0.7366313
5: -1.4758421, -0.6935012, -1.4918607, -0.6933167, -0.3339847, 0.3604457
6: 0.4468049, 0.6884016, 0.4466125, 0.6963563, -0.1101088, 0.1154985
7: -2.4962573, -1.0562958, -2.5304422, -1.0561002, -0.7081559, 0.7258887
8: -4.3944387, -2.7388194, -4.3946891, -2.7365227, -0.9627076, 0.9348202
9: -4.2513075, -2.6350989, -4.2562132, -2.6348186, -1.0716469, 1.0334148

Time for backsubstitution: 6.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 3488
type: B, layer: 1, pos: 228
type: B, layer: 1, pos: 369
type: B, layer: 1, pos: 368
type: B, layer: 1, pos: 307
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 3499
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 579
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 320
type: B, layer: 1, pos: 313
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 3515
type: B, layer: 1, pos: 3519
type: B, layer: 1, pos: 3203
type: B, layer: 1, pos: 594
type: B, layer: 1, pos: 3202
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 341
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3201
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 3094
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3092
type: B, layer: 1, pos: 383
type: B, layer: 1, pos: 2485
type: B, layer: 1, pos: 2376
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 3065
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 3355
type: B, layer: 1, pos: 2453
type: B, layer: 1, pos: 2436
type: B, layer: 1, pos: 3342
type: B, layer: 1, pos: 2452
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 124
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 3098
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2224
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 3099
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 713
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 3379
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 3511
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 2408
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 580
type: B, layer: 1, pos: 2525
type: B, layer: 1, pos: 2260
type: B, layer: 1, pos: 3504
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2982
type: B, layer: 1, pos: 2437
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 2611
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 125
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 3357
type: B, layer: 1, pos: 2825
type: B, layer: 1, pos: 2111
type: B, layer: 1, pos: 3579
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 2583
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 842
type: B, layer: 1, pos: 2318
type: B, layer: 1, pos: 846
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3083
type: B, layer: 1, pos: 3155
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 80
type: B, layer: 1, pos: 2626
type: B, layer: 1, pos: 201
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 2956
type: B, layer: 1, pos: 3005
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 2406
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 860
type: B, layer: 1, pos: 3244
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 694
type: B, layer: 1, pos: 2495
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2486
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2455
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 2475
type: B, layer: 1, pos: 2882
type: B, layer: 1, pos: 873
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 2520
type: B, layer: 1, pos: 2570
type: B, layer: 1, pos: 2416
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 3086
type: B, layer: 1, pos: 2943
type: B, layer: 1, pos: 2929
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 2930
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 2492
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 2506
type: B, layer: 1, pos: 677
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 180
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2477
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3260
type: B, layer: 1, pos: 3327
type: B, layer: 1, pos: 2368
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 841
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 2538
type: B, layer: 1, pos: 3007
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2953
type: B, layer: 1, pos: 3302
type: B, layer: 1, pos: 252
type: B, layer: 1, pos: 3000
type: B, layer: 1, pos: 3249
type: B, layer: 1, pos: 3162
type: B, layer: 1, pos: 3358
type: B, layer: 1, pos: 2410
type: B, layer: 1, pos: 823
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 2252
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2115
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2187
type: B, layer: 1, pos: 2501
type: B, layer: 1, pos: 735
type: B, layer: 1, pos: 3153
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 2535
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 2793
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 838
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 209
type: B, layer: 1, pos: 214
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 222
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 448
type: B, layer: 1, pos: 671
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 896
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 1019
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2519
type: B, layer: 1, pos: 2639
type: B, layer: 1, pos: 2654
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 2698
type: B, layer: 1, pos: 2919
type: B, layer: 1, pos: 2920
type: B, layer: 1, pos: 2939
type: B, layer: 1, pos: 3139
type: B, layer: 1, pos: 3143
type: B, layer: 1, pos: 3147
type: B, layer: 1, pos: 3148
type: B, layer: 1, pos: 3224
type: B, layer: 1, pos: 3359
type: B, layer: 1, pos: 3361
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3596
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 378

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0281050, upper bound: 0.0282916
time: 21.05 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0282855, upper bound: 0.0282938
time: 18.91 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 46.65 seconds
NS_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 46.65
Output dim: 6, lower bound: -0.0268430, upper bound: 0.0275941
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 46.65
Output dim: 6, lower bound: -0.0270265, upper bound: 0.0282917
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 46.65
Output dim: 6, lower bound: -0.0270009, upper bound: 0.0282918
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 46.65
Output dim: 6, lower bound: -0.0271823, upper bound: 0.0282915
NS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 46.65
Output dim: 6, lower bound: -0.0274646, upper bound: 0.0278853
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 46.65
Output dim: 6, lower bound: -0.0276459, upper bound: 0.0282919
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 46.65
Output dim: 6, lower bound: -0.0276195, upper bound: 0.0282913
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 46.65
Output dim: 6, lower bound: -0.0278029, upper bound: 0.0282930
NS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 46.65
Output dim: 6, lower bound: -0.0281025, upper bound: 0.0273698
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 46.65
Output dim: 6, lower bound: -0.0282857, upper bound: 0.0273688
NS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 46.65
Output dim: 6, lower bound: -0.0281051, upper bound: 0.0275912
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 46.65
Output dim: 6, lower bound: -0.0282864, upper bound: 0.0275945
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 46.65
Output dim: 6, lower bound: -0.0273272, upper bound: 0.0282900
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 46.65
Output dim: 6, lower bound: -0.0275097, upper bound: 0.0282918
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 46.65
Output dim: 6, lower bound: -0.0274868, upper bound: 0.0282935
NS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 46.65
Output dim: 6, lower bound: -0.0276683, upper bound: 0.0278853
NS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 46.65
Output dim: 6, lower bound: -0.0281015, upper bound: 0.0278834
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 46.65
Output dim: 6, lower bound: -0.0282854, upper bound: 0.0278877
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 46.65
Output dim: 6, lower bound: -0.0279470, upper bound: 0.0282908
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 46.65
Output dim: 6, lower bound: -0.0281275, upper bound: 0.0282901
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 46.65
Output dim: 6, lower bound: -0.0281050, upper bound: 0.0282916
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 46.65
Output dim: 6, lower bound: -0.0282855, upper bound: 0.0282938

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.9788928, -2.6703677, -3.9885273, -2.6669407, -0.7403607, 0.7420632
1: -4.4265146, -2.4283960, -4.4302425, -2.4271688, -1.4965160, 1.4992800
2: -0.7141430, 0.0171903, -0.7348190, 0.0302228, -0.6727192, 0.6846486
3: -1.1737589, -0.6260446, -1.1848449, -0.6239702, -0.2772186, 0.2884515
4: -0.7824420, 0.1250167, -0.7886226, 0.1391567, -0.7318913, 0.7209901
5: -1.4654170, -0.7083588, -1.4935951, -0.7011082, -0.3267758, 0.3590499
6: 0.4592454, 0.6814554, 0.4538042, 0.6962695, -0.1149303, 0.1018198
7: -2.4811292, -1.0873275, -2.5304604, -1.0805659, -0.6684064, 0.7359962
8: -4.3810291, -2.7571230, -4.3973389, -2.7512887, -0.9342915, 0.9454131
9: -4.2281084, -2.6668308, -4.2533011, -2.6578116, -1.0274531, 1.0485547

Time for backsubstitution: 6.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 3488
type: A, layer: 1, pos: 308
type: A, layer: 1, pos: 369
type: A, layer: 1, pos: 368
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3499
type: A, layer: 1, pos: 309
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 579
type: A, layer: 1, pos: 378
type: A, layer: 1, pos: 320
type: A, layer: 1, pos: 313
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 3515
type: A, layer: 1, pos: 3519
type: A, layer: 1, pos: 3203
type: A, layer: 1, pos: 594
type: A, layer: 1, pos: 3202
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 341
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 3201
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 3094
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3092
type: A, layer: 1, pos: 383
type: A, layer: 1, pos: 2485
type: A, layer: 1, pos: 2376
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 3065
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 3355
type: A, layer: 1, pos: 2453
type: A, layer: 1, pos: 2436
type: A, layer: 1, pos: 3342
type: A, layer: 1, pos: 2452
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 124
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 3098
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 355
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2224
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 3099
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 713
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 3379
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 3511
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 2408
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 580
type: A, layer: 1, pos: 2525
type: A, layer: 1, pos: 2260
type: A, layer: 1, pos: 3504
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2982
type: A, layer: 1, pos: 2437
type: A, layer: 1, pos: 3428
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 2257
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 2611
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 125
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 3357
type: A, layer: 1, pos: 2111
type: A, layer: 1, pos: 2825
type: A, layer: 1, pos: 3579
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 2583
type: A, layer: 1, pos: 842
type: A, layer: 1, pos: 2318
type: A, layer: 1, pos: 846
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 3083
type: A, layer: 1, pos: 3155
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 80
type: A, layer: 1, pos: 2626
type: A, layer: 1, pos: 201
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 2956
type: A, layer: 1, pos: 3005
type: A, layer: 1, pos: 2209
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 2406
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 860
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 3244
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 694
type: A, layer: 1, pos: 2495
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2486
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2455
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 3245
type: A, layer: 1, pos: 2475
type: A, layer: 1, pos: 2882
type: A, layer: 1, pos: 873
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 2520
type: A, layer: 1, pos: 2570
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2416
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 3086
type: A, layer: 1, pos: 2943
type: A, layer: 1, pos: 2929
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 2930
type: A, layer: 1, pos: 2492
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 2506
type: A, layer: 1, pos: 677
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 180
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2477
type: A, layer: 1, pos: 3260
type: A, layer: 1, pos: 3327
type: A, layer: 1, pos: 2368
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 2538
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 841
type: A, layer: 1, pos: 3007
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2953
type: A, layer: 1, pos: 252
type: A, layer: 1, pos: 3302
type: A, layer: 1, pos: 3000
type: A, layer: 1, pos: 3162
type: A, layer: 1, pos: 3249
type: A, layer: 1, pos: 3358
type: A, layer: 1, pos: 2410
type: A, layer: 1, pos: 823
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 2252
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2115
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2187
type: A, layer: 1, pos: 2501
type: A, layer: 1, pos: 735
type: A, layer: 1, pos: 3153
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 2535
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 855
type: A, layer: 1, pos: 2793
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 209
type: A, layer: 1, pos: 214
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 222
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 448
type: A, layer: 1, pos: 671
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 896
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 1019
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2519
type: A, layer: 1, pos: 2639
type: A, layer: 1, pos: 2654
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 2698
type: A, layer: 1, pos: 2919
type: A, layer: 1, pos: 2920
type: A, layer: 1, pos: 2939
type: A, layer: 1, pos: 3139
type: A, layer: 1, pos: 3143
type: A, layer: 1, pos: 3147
type: A, layer: 1, pos: 3148
type: A, layer: 1, pos: 3224
type: A, layer: 1, pos: 3359
type: A, layer: 1, pos: 3361
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3596
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 367

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0269789, upper bound: 0.0280477
time: 73.37 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0270244, upper bound: 0.0282898
time: 102.01 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 130.58 + 3501.01 = 3631.60 seconds
