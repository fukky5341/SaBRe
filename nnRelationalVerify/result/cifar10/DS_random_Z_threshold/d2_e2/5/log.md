## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 5)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.2434052511


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.0364926, -1.6305044, -3.0364926, -1.6305044, -1.0001023, 1.0001023)
1: (-2.3519323, -1.2734849, -2.3519323, -1.2734849, -0.5936676, 0.5936675)
2: (-1.0463166, -0.0026524, -1.0463166, -0.0026524, -0.8867217, 0.8867217)
3: (-0.4830326, 0.1667036, -0.4830326, 0.1667036, -0.6032193, 0.6032193)
4: (-1.9464408, -0.7899504, -1.9464408, -0.7899504, -0.6373334, 0.6373334)
5: (-0.6560865, 0.0364869, -0.6560865, 0.0364869, -0.6329277, 0.6329277)
6: (-1.0628040, -0.2340931, -1.0628040, -0.2340931, -0.5147617, 0.5147617)
7: (-2.2468164, -0.7752080, -2.2468164, -0.7752080, -0.8829843, 0.8829844)
8: (-1.3536971, -0.6113571, -1.3536971, -0.6113571, -0.3843989, 0.3843988)
9: (-1.1448197, -0.3270659, -1.1448197, -0.3270659, -0.4200414, 0.4200414)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.07 + 89.91 = 96.98 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.2436462, upper bound: 0.2436487

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 2815
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 3187
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 3429
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2668

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 129

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2435800, upper bound: 0.2435846
time: 9.27 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2435808, upper bound: 0.2435849
time: 9.07 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 18.36 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 18.36
Output dim: 3, lower bound: -0.2435800, upper bound: 0.2435846
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 18.36
Output dim: 3, lower bound: -0.2435808, upper bound: 0.2435849

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.0364926, -1.6305044, -3.0364926, -1.6305044, -1.0000892, 1.0000894
1: -2.3519323, -1.2734849, -2.3519323, -1.2734849, -0.5936528, 0.5936530
2: -1.0463166, -0.0026524, -1.0463166, -0.0026524, -0.8867690, 0.8867671
3: -0.4830326, 0.1667036, -0.4830326, 0.1667036, -0.6032178, 0.6032176
4: -1.9464408, -0.7899504, -1.9464408, -0.7899504, -0.6373308, 0.6373310
5: -0.6560865, 0.0364869, -0.6560865, 0.0364869, -0.6329075, 0.6329069
6: -1.0628040, -0.2340931, -1.0628040, -0.2340931, -0.5147442, 0.5147443
7: -2.2468164, -0.7752080, -2.2468164, -0.7752080, -0.8829789, 0.8829789
8: -1.3536971, -0.6113571, -1.3536971, -0.6113571, -0.3843983, 0.3843983
9: -1.1448197, -0.3270659, -1.1448197, -0.3270659, -0.4200400, 0.4200402

Time for backsubstitution: 5.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2815
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 3429
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3187
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2474

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 513

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2435752, upper bound: 0.2435526
time: 450.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2435545, upper bound: 0.2435778
time: 308.37 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.0364926, -1.6305044, -3.0364926, -1.6305044, -1.0000895, 1.0000892
1: -2.3519323, -1.2734849, -2.3519323, -1.2734849, -0.5936531, 0.5936528
2: -1.0463166, -0.0026524, -1.0463166, -0.0026524, -0.8867671, 0.8867691
3: -0.4830326, 0.1667036, -0.4830326, 0.1667036, -0.6032177, 0.6032178
4: -1.9464408, -0.7899504, -1.9464408, -0.7899504, -0.6373310, 0.6373308
5: -0.6560865, 0.0364869, -0.6560865, 0.0364869, -0.6329069, 0.6329075
6: -1.0628040, -0.2340931, -1.0628040, -0.2340931, -0.5147444, 0.5147441
7: -2.2468164, -0.7752080, -2.2468164, -0.7752080, -0.8829789, 0.8829789
8: -1.3536971, -0.6113571, -1.3536971, -0.6113571, -0.3843983, 0.3843983
9: -1.1448197, -0.3270659, -1.1448197, -0.3270659, -0.4200400, 0.4200402

Time for backsubstitution: 5.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2815
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 3429
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 3187
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 731

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3021

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2434809, upper bound: 0.2434456
time: 506.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2434427, upper bound: 0.2434834
time: 450.66 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 963.05 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 963.05
Output dim: 3, lower bound: -0.2435752, upper bound: 0.2435526
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 963.05
Output dim: 3, lower bound: -0.2435545, upper bound: 0.2435778
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 963.05
Output dim: 3, lower bound: -0.2434809, upper bound: 0.2434456
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 963.05
Output dim: 3, lower bound: -0.2434427, upper bound: 0.2434834

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0364926, -1.6305044, -3.0364926, -1.6305044, -1.0000334, 1.0000323
1: -2.3519323, -1.2734849, -2.3519323, -1.2734849, -0.5936316, 0.5936317
2: -1.0463166, -0.0026524, -1.0463166, -0.0026524, -0.8867701, 0.8867671
3: -0.4830326, 0.1667036, -0.4830326, 0.1667036, -0.6032137, 0.6032115
4: -1.9464408, -0.7899504, -1.9464408, -0.7899504, -0.6370069, 0.6370401
5: -0.6560865, 0.0364869, -0.6560865, 0.0364869, -0.6329075, 0.6329069
6: -1.0628040, -0.2340931, -1.0628040, -0.2340931, -0.5147432, 0.5147436
7: -2.2468164, -0.7752080, -2.2468164, -0.7752080, -0.8829724, 0.8829726
8: -1.3536971, -0.6113571, -1.3536971, -0.6113571, -0.3843958, 0.3843953
9: -1.1448197, -0.3270659, -1.1448197, -0.3270659, -0.4198228, 0.4198099

Time for backsubstitution: 5.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 3429
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3187
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2815
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 2600

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3036

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2434187, upper bound: 0.2434778
time: 524.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2434984, upper bound: 0.2434014
time: 13.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.0364926, -1.6305044, -3.0364926, -1.6305044, -1.0000321, 1.0000336
1: -2.3519323, -1.2734849, -2.3519323, -1.2734849, -0.5936313, 0.5936320
2: -1.0463166, -0.0026524, -1.0463166, -0.0026524, -0.8867691, 0.8867682
3: -0.4830326, 0.1667036, -0.4830326, 0.1667036, -0.6032116, 0.6032135
4: -1.9464408, -0.7899504, -1.9464408, -0.7899504, -0.6370399, 0.6370071
5: -0.6560865, 0.0364869, -0.6560865, 0.0364869, -0.6329075, 0.6329069
6: -1.0628040, -0.2340931, -1.0628040, -0.2340931, -0.5147433, 0.5147436
7: -2.2468164, -0.7752080, -2.2468164, -0.7752080, -0.8829726, 0.8829725
8: -1.3536971, -0.6113571, -1.3536971, -0.6113571, -0.3843952, 0.3843959
9: -1.1448197, -0.3270659, -1.1448197, -0.3270659, -0.4198099, 0.4198228

Time for backsubstitution: 5.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3429
type: DSZ, layer: 1, pos: 2815
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 3187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 72

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2435478, upper bound: 0.2435742
time: 17.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2435507, upper bound: 0.2435697
time: 20.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.0364926, -1.6305044, -3.0364926, -1.6305044, -0.9996088, 0.9996653
1: -2.3519323, -1.2734849, -2.3519323, -1.2734849, -0.5934361, 0.5933194
2: -1.0463166, -0.0026524, -1.0463166, -0.0026524, -0.8867409, 0.8867395
3: -0.4830326, 0.1667036, -0.4830326, 0.1667036, -0.6032172, 0.6032175
4: -1.9464408, -0.7899504, -1.9464408, -0.7899504, -0.6373301, 0.6373298
5: -0.6560865, 0.0364869, -0.6560865, 0.0364869, -0.6329062, 0.6329066
6: -1.0628040, -0.2340931, -1.0628040, -0.2340931, -0.5146503, 0.5146502
7: -2.2468164, -0.7752080, -2.2468164, -0.7752080, -0.8829723, 0.8829695
8: -1.3536971, -0.6113571, -1.3536971, -0.6113571, -0.3842404, 0.3842857
9: -1.1448197, -0.3270659, -1.1448197, -0.3270659, -0.4199844, 0.4199753

Time for backsubstitution: 5.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 3429
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 3187
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 2815
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3056

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3319

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2434395, upper bound: 0.2434313
time: 84.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2434698, upper bound: 0.2434013
time: 689.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.0364926, -1.6305044, -3.0364926, -1.6305044, -0.9996657, 0.9996083
1: -2.3519323, -1.2734849, -2.3519323, -1.2734849, -0.5933196, 0.5934358
2: -1.0463166, -0.0026524, -1.0463166, -0.0026524, -0.8867375, 0.8867428
3: -0.4830326, 0.1667036, -0.4830326, 0.1667036, -0.6032174, 0.6032173
4: -1.9464408, -0.7899504, -1.9464408, -0.7899504, -0.6373301, 0.6373299
5: -0.6560865, 0.0364869, -0.6560865, 0.0364869, -0.6329060, 0.6329067
6: -1.0628040, -0.2340931, -1.0628040, -0.2340931, -0.5146506, 0.5146500
7: -2.2468164, -0.7752080, -2.2468164, -0.7752080, -0.8829694, 0.8829724
8: -1.3536971, -0.6113571, -1.3536971, -0.6113571, -0.3842858, 0.3842403
9: -1.1448197, -0.3270659, -1.1448197, -0.3270659, -0.4199753, 0.4199844

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2815
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 3187
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 3429
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 3032

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2630

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2433980, upper bound: 0.2433087
time: 9.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2432644, upper bound: 0.2434398
time: 215.02 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 230.58 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 230.58
Output dim: 3, lower bound: -0.2434187, upper bound: 0.2434778
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 230.58
Output dim: 3, lower bound: -0.2434984, upper bound: 0.2434014
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 230.58
Output dim: 3, lower bound: -0.2435478, upper bound: 0.2435742
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 230.58
Output dim: 3, lower bound: -0.2435507, upper bound: 0.2435697
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 230.58
Output dim: 3, lower bound: -0.2434395, upper bound: 0.2434313
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 230.58
Output dim: 3, lower bound: -0.2434698, upper bound: 0.2434013
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 230.58
Output dim: 3, lower bound: -0.2433980, upper bound: 0.2433087
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 230.58
Output dim: 3, lower bound: -0.2432644, upper bound: 0.2434398

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0364926, -1.6305044, -3.0364926, -1.6305044, -0.9976020, 0.9972610
1: -2.3519323, -1.2734849, -2.3519323, -1.2734849, -0.5921740, 0.5918317
2: -1.0463166, -0.0026524, -1.0463166, -0.0026524, -0.8866543, 0.8866310
3: -0.4830326, 0.1667036, -0.4830326, 0.1667036, -0.6032076, 0.6032062
4: -1.9464408, -0.7899504, -1.9464408, -0.7899504, -0.6368489, 0.6368924
5: -0.6560865, 0.0364869, -0.6560865, 0.0364869, -0.6328987, 0.6328983
6: -1.0628040, -0.2340931, -1.0628040, -0.2340931, -0.5143005, 0.5143725
7: -2.2468164, -0.7752080, -2.2468164, -0.7752080, -0.8829715, 0.8829716
8: -1.3536971, -0.6113571, -1.3536971, -0.6113571, -0.3838460, 0.3837449
9: -1.1448197, -0.3270659, -1.1448197, -0.3270659, -0.4195803, 0.4195216

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2815
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 3457
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3187
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3429
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2834

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3044

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2434153, upper bound: 0.2434795
time: 11.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2434153, upper bound: 0.2434782
time: 223.95 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 241.11 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 241.11
Output dim: 3, lower bound: -0.2434153, upper bound: 0.2434795
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 241.11
Output dim: 3, lower bound: -0.2434153, upper bound: 0.2434782
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 241.11
Output dim: 3, lower bound: -0.2434984, upper bound: 0.2434014
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 241.11
Output dim: 3, lower bound: -0.2435478, upper bound: 0.2435742
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 241.11
Output dim: 3, lower bound: -0.2435507, upper bound: 0.2435697
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 241.11
Output dim: 3, lower bound: -0.2434395, upper bound: 0.2434313
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 241.11
Output dim: 3, lower bound: -0.2434698, upper bound: 0.2434013
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 241.11
Output dim: 3, lower bound: -0.2432644, upper bound: 0.2434398

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 96.98 + 3585.97 = 3682.95 seconds
