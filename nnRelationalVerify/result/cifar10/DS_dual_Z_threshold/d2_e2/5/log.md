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
execution time: IAR + RelationalAnalysis = 8.16 + 90.91 = 99.07 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.2436462, upper bound: 0.2436487

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 426
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2815
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3187
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 3429
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3457

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 426

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2436077, upper bound: 0.2434085
time: 889.81 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2434094, upper bound: 0.2436077
time: 218.82 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1108.71 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1108.71
Output dim: 3, lower bound: -0.2436077, upper bound: 0.2434085
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1108.71
Output dim: 3, lower bound: -0.2434094, upper bound: 0.2436077

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.0364926, -1.6305044, -3.0364926, -1.6305044, -0.9998104, 0.9997369
1: -2.3519323, -1.2734849, -2.3519323, -1.2734849, -0.5934127, 0.5933437
2: -1.0463166, -0.0026524, -1.0463166, -0.0026524, -0.8866819, 0.8866857
3: -0.4830326, 0.1667036, -0.4830326, 0.1667036, -0.6031954, 0.6031981
4: -1.9464408, -0.7899504, -1.9464408, -0.7899504, -0.6373310, 0.6373315
5: -0.6560865, 0.0364869, -0.6560865, 0.0364869, -0.6329095, 0.6329135
6: -1.0628040, -0.2340931, -1.0628040, -0.2340931, -0.5147361, 0.5147337
7: -2.2468164, -0.7752080, -2.2468164, -0.7752080, -0.8829812, 0.8829820
8: -1.3536971, -0.6113571, -1.3536971, -0.6113571, -0.3842615, 0.3842304
9: -1.1448197, -0.3270659, -1.1448197, -0.3270659, -0.4199666, 0.4199436

Time for backsubstitution: 6.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2815
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3187
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 3429
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3457

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3070

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2434152, upper bound: 0.2434037
time: 9.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2436017, upper bound: 0.2432153
time: 528.45 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.0364926, -1.6305044, -3.0364926, -1.6305044, -0.9997367, 0.9998106
1: -2.3519323, -1.2734849, -2.3519323, -1.2734849, -0.5933437, 0.5934126
2: -1.0463166, -0.0026524, -1.0463166, -0.0026524, -0.8866857, 0.8866819
3: -0.4830326, 0.1667036, -0.4830326, 0.1667036, -0.6031981, 0.6031954
4: -1.9464408, -0.7899504, -1.9464408, -0.7899504, -0.6373315, 0.6373310
5: -0.6560865, 0.0364869, -0.6560865, 0.0364869, -0.6329135, 0.6329095
6: -1.0628040, -0.2340931, -1.0628040, -0.2340931, -0.5147337, 0.5147361
7: -2.2468164, -0.7752080, -2.2468164, -0.7752080, -0.8829820, 0.8829812
8: -1.3536971, -0.6113571, -1.3536971, -0.6113571, -0.3842305, 0.3842615
9: -1.1448197, -0.3270659, -1.1448197, -0.3270659, -0.4199435, 0.4199668

Time for backsubstitution: 5.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2815
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3187
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 3429
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3457

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3070

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2432164, upper bound: 0.2436039
time: 223.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2434026, upper bound: 0.2434148
time: 102.07 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 330.98 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 330.98
Output dim: 3, lower bound: -0.2434152, upper bound: 0.2434037
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 330.98
Output dim: 3, lower bound: -0.2436017, upper bound: 0.2432153
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 330.98
Output dim: 3, lower bound: -0.2432164, upper bound: 0.2436039
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 330.98
Output dim: 3, lower bound: -0.2434026, upper bound: 0.2434148

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0364926, -1.6305044, -3.0364926, -1.6305044, -0.9996618, 0.9995613
1: -2.3519323, -1.2734849, -2.3519323, -1.2734849, -0.5932820, 0.5931919
2: -1.0463166, -0.0026524, -1.0463166, -0.0026524, -0.8866801, 0.8866812
3: -0.4830326, 0.1667036, -0.4830326, 0.1667036, -0.6031950, 0.6031977
4: -1.9464408, -0.7899504, -1.9464408, -0.7899504, -0.6373273, 0.6373303
5: -0.6560865, 0.0364869, -0.6560865, 0.0364869, -0.6329082, 0.6329125
6: -1.0628040, -0.2340931, -1.0628040, -0.2340931, -0.5147344, 0.5147321
7: -2.2468164, -0.7752080, -2.2468164, -0.7752080, -0.8829664, 0.8829682
8: -1.3536971, -0.6113571, -1.3536971, -0.6113571, -0.3841697, 0.3841295
9: -1.1448197, -0.3270659, -1.1448197, -0.3270659, -0.4199488, 0.4199209

Time for backsubstitution: 5.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2815
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3187
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 3429
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3457

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2394

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2432628, upper bound: 0.2433281
time: 54.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2433399, upper bound: 0.2432511
time: 218.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.0364926, -1.6305044, -3.0364926, -1.6305044, -0.9996349, 0.9995883
1: -2.3519323, -1.2734849, -2.3519323, -1.2734849, -0.5932608, 0.5932131
2: -1.0463166, -0.0026524, -1.0463166, -0.0026524, -0.8866776, 0.8866838
3: -0.4830326, 0.1667036, -0.4830326, 0.1667036, -0.6031950, 0.6031978
4: -1.9464408, -0.7899504, -1.9464408, -0.7899504, -0.6373297, 0.6373279
5: -0.6560865, 0.0364869, -0.6560865, 0.0364869, -0.6329085, 0.6329123
6: -1.0628040, -0.2340931, -1.0628040, -0.2340931, -0.5147346, 0.5147320
7: -2.2468164, -0.7752080, -2.2468164, -0.7752080, -0.8829675, 0.8829672
8: -1.3536971, -0.6113571, -1.3536971, -0.6113571, -0.3841605, 0.3841387
9: -1.1448197, -0.3270659, -1.1448197, -0.3270659, -0.4199441, 0.4199255

Time for backsubstitution: 5.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2815
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3187
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 3429
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3457

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2394

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2434484, upper bound: 0.2431443
time: 99.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2435263, upper bound: 0.2430631
time: 40.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.0364926, -1.6305044, -3.0364926, -1.6305044, -0.9995881, 0.9996350
1: -2.3519323, -1.2734849, -2.3519323, -1.2734849, -0.5932130, 0.5932608
2: -1.0463166, -0.0026524, -1.0463166, -0.0026524, -0.8866839, 0.8866775
3: -0.4830326, 0.1667036, -0.4830326, 0.1667036, -0.6031977, 0.6031950
4: -1.9464408, -0.7899504, -1.9464408, -0.7899504, -0.6373278, 0.6373298
5: -0.6560865, 0.0364869, -0.6560865, 0.0364869, -0.6329123, 0.6329085
6: -1.0628040, -0.2340931, -1.0628040, -0.2340931, -0.5147321, 0.5147344
7: -2.2468164, -0.7752080, -2.2468164, -0.7752080, -0.8829671, 0.8829675
8: -1.3536971, -0.6113571, -1.3536971, -0.6113571, -0.3841387, 0.3841605
9: -1.1448197, -0.3270659, -1.1448197, -0.3270659, -0.4199256, 0.4199440

Time for backsubstitution: 5.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2815
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3187
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 3429
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3457

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2394

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2430649, upper bound: 0.2435266
time: 11.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2431418, upper bound: 0.2434527
time: 75.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.0364926, -1.6305044, -3.0364926, -1.6305044, -0.9995612, 0.9996619
1: -2.3519323, -1.2734849, -2.3519323, -1.2734849, -0.5931918, 0.5932820
2: -1.0463166, -0.0026524, -1.0463166, -0.0026524, -0.8866813, 0.8866800
3: -0.4830326, 0.1667036, -0.4830326, 0.1667036, -0.6031977, 0.6031950
4: -1.9464408, -0.7899504, -1.9464408, -0.7899504, -0.6373302, 0.6373274
5: -0.6560865, 0.0364869, -0.6560865, 0.0364869, -0.6329125, 0.6329082
6: -1.0628040, -0.2340931, -1.0628040, -0.2340931, -0.5147322, 0.5147344
7: -2.2468164, -0.7752080, -2.2468164, -0.7752080, -0.8829682, 0.8829665
8: -1.3536971, -0.6113571, -1.3536971, -0.6113571, -0.3841295, 0.3841697
9: -1.1448197, -0.3270659, -1.1448197, -0.3270659, -0.4199210, 0.4199486

Time for backsubstitution: 5.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2394
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2815
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3187
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 3429
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3457

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2394

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2432513, upper bound: 0.2433440
time: 19.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2433299, upper bound: 0.2432641
time: 66.92 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 91.45 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 91.45
Output dim: 3, lower bound: -0.2432628, upper bound: 0.2433281
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 91.45
Output dim: 3, lower bound: -0.2433399, upper bound: 0.2432511
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 91.45
Output dim: 3, lower bound: -0.2434484, upper bound: 0.2431443
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 91.45
Output dim: 3, lower bound: -0.2435263, upper bound: 0.2430631
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 91.45
Output dim: 3, lower bound: -0.2430649, upper bound: 0.2435266
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 91.45
Output dim: 3, lower bound: -0.2431418, upper bound: 0.2434527
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 91.45
Output dim: 3, lower bound: -0.2432513, upper bound: 0.2433440
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 91.45
Output dim: 3, lower bound: -0.2433299, upper bound: 0.2432641

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.0364926, -1.6305044, -3.0364926, -1.6305044, -0.9996341, 0.9995874
1: -2.3519323, -1.2734849, -2.3519323, -1.2734849, -0.5932599, 0.5932119
2: -1.0463166, -0.0026524, -1.0463166, -0.0026524, -0.8866774, 0.8866838
3: -0.4830326, 0.1667036, -0.4830326, 0.1667036, -0.6031950, 0.6031978
4: -1.9464408, -0.7899504, -1.9464408, -0.7899504, -0.6373286, 0.6373270
5: -0.6560865, 0.0364869, -0.6560865, 0.0364869, -0.6329085, 0.6329123
6: -1.0628040, -0.2340931, -1.0628040, -0.2340931, -0.5147319, 0.5147303
7: -2.2468164, -0.7752080, -2.2468164, -0.7752080, -0.8829635, 0.8829632
8: -1.3536971, -0.6113571, -1.3536971, -0.6113571, -0.3841602, 0.3841385
9: -1.1448197, -0.3270659, -1.1448197, -0.3270659, -0.4199432, 0.4199246

Time for backsubstitution: 5.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2815
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3187
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 3429
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3457

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2380

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2433098, upper bound: 0.2431315
time: 36.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2434433, upper bound: 0.2430145
time: 73.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.0364926, -1.6305044, -3.0364926, -1.6305044, -0.9996340, 0.9995874
1: -2.3519323, -1.2734849, -2.3519323, -1.2734849, -0.5932598, 0.5932121
2: -1.0463166, -0.0026524, -1.0463166, -0.0026524, -0.8866774, 0.8866837
3: -0.4830326, 0.1667036, -0.4830326, 0.1667036, -0.6031950, 0.6031978
4: -1.9464408, -0.7899504, -1.9464408, -0.7899504, -0.6373288, 0.6373267
5: -0.6560865, 0.0364869, -0.6560865, 0.0364869, -0.6329085, 0.6329123
6: -1.0628040, -0.2340931, -1.0628040, -0.2340931, -0.5147329, 0.5147294
7: -2.2468164, -0.7752080, -2.2468164, -0.7752080, -0.8829634, 0.8829632
8: -1.3536971, -0.6113571, -1.3536971, -0.6113571, -0.3841602, 0.3841384
9: -1.1448197, -0.3270659, -1.1448197, -0.3270659, -0.4199429, 0.4199247

Time for backsubstitution: 6.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2815
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3187
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 3429
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3457

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2380

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2433950, upper bound: 0.2430594
time: 305.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2435154, upper bound: 0.2429277
time: 134.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.0364926, -1.6305044, -3.0364926, -1.6305044, -0.9995874, 0.9996341
1: -2.3519323, -1.2734849, -2.3519323, -1.2734849, -0.5932121, 0.5932598
2: -1.0463166, -0.0026524, -1.0463166, -0.0026524, -0.8866838, 0.8866774
3: -0.4830326, 0.1667036, -0.4830326, 0.1667036, -0.6031977, 0.6031950
4: -1.9464408, -0.7899504, -1.9464408, -0.7899504, -0.6373267, 0.6373289
5: -0.6560865, 0.0364869, -0.6560865, 0.0364869, -0.6329123, 0.6329085
6: -1.0628040, -0.2340931, -1.0628040, -0.2340931, -0.5147296, 0.5147328
7: -2.2468164, -0.7752080, -2.2468164, -0.7752080, -0.8829631, 0.8829634
8: -1.3536971, -0.6113571, -1.3536971, -0.6113571, -0.3841384, 0.3841602
9: -1.1448197, -0.3270659, -1.1448197, -0.3270659, -0.4199246, 0.4199430

Time for backsubstitution: 5.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2815
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3187
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 3429
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3457

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2380

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2429265, upper bound: 0.2435154
time: 37.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2430588, upper bound: 0.2433944
time: 110.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.0364926, -1.6305044, -3.0364926, -1.6305044, -0.9995873, 0.9996341
1: -2.3519323, -1.2734849, -2.3519323, -1.2734849, -0.5932120, 0.5932598
2: -1.0463166, -0.0026524, -1.0463166, -0.0026524, -0.8866838, 0.8866774
3: -0.4830326, 0.1667036, -0.4830326, 0.1667036, -0.6031977, 0.6031950
4: -1.9464408, -0.7899504, -1.9464408, -0.7899504, -0.6373269, 0.6373286
5: -0.6560865, 0.0364869, -0.6560865, 0.0364869, -0.6329123, 0.6329085
6: -1.0628040, -0.2340931, -1.0628040, -0.2340931, -0.5147304, 0.5147319
7: -2.2468164, -0.7752080, -2.2468164, -0.7752080, -0.8829631, 0.8829634
8: -1.3536971, -0.6113571, -1.3536971, -0.6113571, -0.3841385, 0.3841602
9: -1.1448197, -0.3270659, -1.1448197, -0.3270659, -0.4199244, 0.4199433

Time for backsubstitution: 5.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2380
type: DSZ, layer: 1, pos: 3056
type: DSZ, layer: 1, pos: 427
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 3100
type: DSZ, layer: 1, pos: 2423
type: DSZ, layer: 1, pos: 2139
type: DSZ, layer: 1, pos: 3057
type: DSZ, layer: 1, pos: 3038
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2815
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 413
type: DSZ, layer: 1, pos: 2352
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3037
type: DSZ, layer: 1, pos: 3036
type: DSZ, layer: 1, pos: 3097
type: DSZ, layer: 1, pos: 2873
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2422
type: DSZ, layer: 1, pos: 3263
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3565
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3321
type: DSZ, layer: 1, pos: 3319
type: DSZ, layer: 1, pos: 3267
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 597
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 295
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 2642
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3018
type: DSZ, layer: 1, pos: 3320
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2417
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2402
type: DSZ, layer: 1, pos: 2854
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2584
type: DSZ, layer: 1, pos: 2630
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 3275
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 150
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 200
type: DSZ, layer: 1, pos: 239
type: DSZ, layer: 1, pos: 279
type: DSZ, layer: 1, pos: 423
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 751
type: DSZ, layer: 1, pos: 824
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2079
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2250
type: DSZ, layer: 1, pos: 2317
type: DSZ, layer: 1, pos: 2326
type: DSZ, layer: 1, pos: 2327
type: DSZ, layer: 1, pos: 2328
type: DSZ, layer: 1, pos: 2354
type: DSZ, layer: 1, pos: 2400
type: DSZ, layer: 1, pos: 2414
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2461
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2489
type: DSZ, layer: 1, pos: 2511
type: DSZ, layer: 1, pos: 2522
type: DSZ, layer: 1, pos: 2526
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2668
type: DSZ, layer: 1, pos: 2753
type: DSZ, layer: 1, pos: 2791
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2863
type: DSZ, layer: 1, pos: 2866
type: DSZ, layer: 1, pos: 2926
type: DSZ, layer: 1, pos: 2991
type: DSZ, layer: 1, pos: 2992
type: DSZ, layer: 1, pos: 3008
type: DSZ, layer: 1, pos: 3044
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3074
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3090
type: DSZ, layer: 1, pos: 3104
type: DSZ, layer: 1, pos: 3187
type: DSZ, layer: 1, pos: 3232
type: DSZ, layer: 1, pos: 3269
type: DSZ, layer: 1, pos: 3271
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 3429
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3457

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2380

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2430113, upper bound: 0.2434462
time: 15.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2431294, upper bound: 0.2433113
time: 294.41 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 315.93 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 315.93
Output dim: 3, lower bound: -0.2433098, upper bound: 0.2431315
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 315.93
Output dim: 3, lower bound: -0.2434433, upper bound: 0.2430145
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 315.93
Output dim: 3, lower bound: -0.2433950, upper bound: 0.2430594
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 315.93
Output dim: 3, lower bound: -0.2435154, upper bound: 0.2429277
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 315.93
Output dim: 3, lower bound: -0.2429265, upper bound: 0.2435154
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 315.93
Output dim: 3, lower bound: -0.2430588, upper bound: 0.2433944
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 315.93
Output dim: 3, lower bound: -0.2430113, upper bound: 0.2434462
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 315.93
Output dim: 3, lower bound: -0.2431294, upper bound: 0.2433113

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 99.07 + 3625.04 = 3724.11 seconds
