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
execution time: IAR + RelationalAnalysis = 7.81 + 91.09 = 98.89 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.2436462, upper bound: 0.2436487

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 413
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3036
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 3037
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2854
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2394
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 3441
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 3429
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 3259
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 3321
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 3423
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2815
type: A, layer: 1, pos: 3320
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 3275
type: A, layer: 1, pos: 2753
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3452
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 3436
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 3232
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3271
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2791
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 3422
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 279
type: A, layer: 1, pos: 3564
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2461
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2489
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3269

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 413

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2436150, upper bound: 0.2432890
time: 66.55 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2436406, upper bound: 0.2436401
time: 89.17 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 155.80 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 155.80
Output dim: 3, lower bound: -0.2436150, upper bound: 0.2432890
NS_A2, status: Status.UNKNOWN, split count: 1, time: 155.80
Output dim: 3, lower bound: -0.2436406, upper bound: 0.2436401

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3.0323732, -1.6372790, -3.0348463, -1.6357672, -0.9902782, 0.9912055
1: -2.3496599, -1.2733897, -2.3501611, -1.2738550, -0.5902694, 0.5907195
2: -1.0440280, -0.0097770, -1.0456637, -0.0086753, -0.8786077, 0.8790932
3: -0.4808500, 0.1630810, -0.4825196, 0.1637162, -0.5979146, 0.5989082
4: -1.9480153, -0.7913182, -1.9457490, -0.7909187, -0.6334012, 0.6333248
5: -0.6538635, 0.0327829, -0.6553826, 0.0335021, -0.6274831, 0.6282517
6: -1.0617363, -0.2351766, -1.0626856, -0.2343358, -0.5130503, 0.5136419
7: -2.2450361, -0.7770126, -2.2449043, -0.7767158, -0.8778675, 0.8778143
8: -1.3525968, -0.6118833, -1.3528147, -0.6118101, -0.3825001, 0.3823222
9: -1.1437751, -0.3275126, -1.1441867, -0.3272710, -0.4189246, 0.4192545

Time for backsubstitution: 6.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 3037
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2854
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2394
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 413
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 3441
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 2853
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 3429
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 3263
type: B, layer: 1, pos: 3321
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3423
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2815
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 2753
type: B, layer: 1, pos: 3319
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3452
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 3436
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 2400
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3232
type: B, layer: 1, pos: 3271
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2771
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2791
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 3422
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 279
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2461
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3269

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2600

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2433707, upper bound: 0.2432190
time: 169.56 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2435444, upper bound: 0.2432188
time: 152.79 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -3.0364776, -1.6305404, -3.0364804, -1.6305318, -1.0001085, 1.0000560
1: -2.3508179, -1.2734864, -2.3510592, -1.2734860, -0.5920969, 0.5928344
2: -1.0463058, -0.0026722, -1.0463078, -0.0026675, -0.8866929, 0.8840918
3: -0.4830285, 0.1666961, -0.4830292, 0.1666979, -0.6032099, 0.6020762
4: -1.9464359, -0.7908475, -1.9464365, -0.7906609, -0.6380078, 0.6368018
5: -0.6560822, 0.0364836, -0.6560832, 0.0364844, -0.6329210, 0.6323040
6: -1.0628036, -0.2340940, -1.0628035, -0.2340938, -0.5143576, 0.5144082
7: -2.2468033, -0.7756599, -2.2468066, -0.7755665, -0.8829759, 0.8829691
8: -1.3535025, -0.6113594, -1.3535483, -0.6113589, -0.3831993, 0.3843884
9: -1.1443183, -0.3270667, -1.1444291, -0.3270666, -0.4196140, 0.4195015

Time for backsubstitution: 6.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 3037
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2854
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2394
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 413
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 3441
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 2853
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 3429
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 3263
type: B, layer: 1, pos: 3321
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3423
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2815
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 2753
type: B, layer: 1, pos: 3319
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3452
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 3436
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 2400
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3232
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3271
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2771
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2791
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 3422
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 279
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2461
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3269

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2600

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2433973, upper bound: 0.2435708
time: 10.04 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2435704, upper bound: 0.2435743
time: 10.26 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 26.76 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 26.76
Output dim: 3, lower bound: -0.2433707, upper bound: 0.2432190
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 26.76
Output dim: 3, lower bound: -0.2435444, upper bound: 0.2432188
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 26.76
Output dim: 3, lower bound: -0.2433973, upper bound: 0.2435708
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 26.76
Output dim: 3, lower bound: -0.2435704, upper bound: 0.2435743

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -3.0301912, -1.6372812, -3.0324686, -1.6289883, -0.9964294, 0.9882156
1: -2.3461380, -1.2733933, -2.3461719, -1.2697718, -0.5927421, 0.5862771
2: -1.0439579, -0.0098349, -1.0462973, -0.0087482, -0.8781777, 0.8798637
3: -0.4807787, 0.1629039, -0.4877609, 0.1635401, -0.5976358, 0.6039492
4: -1.9479680, -0.7917497, -1.9478668, -0.7914481, -0.6328088, 0.6350588
5: -0.6537901, 0.0326336, -0.6607106, 0.0333427, -0.6272156, 0.6333770
6: -1.0617166, -0.2352054, -1.0655078, -0.2343552, -0.5130382, 0.5160626
7: -2.2449346, -0.7771876, -2.2495341, -0.7769274, -0.8776237, 0.8822918
8: -1.3518634, -0.6119187, -1.3519368, -0.6094230, -0.3849163, 0.3812294
9: -1.1421757, -0.3275130, -1.1424069, -0.3269702, -0.4180781, 0.4174738

Time for backsubstitution: 6.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3036
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 3037
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2854
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2394
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 3441
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 3429
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 3259
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 3321
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 3423
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2815
type: A, layer: 1, pos: 3320
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 3275
type: A, layer: 1, pos: 2753
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3452
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 3436
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 3271
type: A, layer: 1, pos: 3232
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2791
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 3422
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 279
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 3564
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2461
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2489
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3269

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3100

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2433676, upper bound: 0.2431943
time: 278.23 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2435205, upper bound: 0.2431936
time: 37.63 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -3.0341473, -1.6305418, -3.0334823, -1.6305337, -0.9975112, 0.9968637
1: -2.3478065, -1.2734904, -2.3472030, -1.2734909, -0.5887560, 0.5886626
2: -1.0462525, -0.0027110, -1.0462397, -0.0027177, -0.8864499, 0.8838573
3: -0.4829547, 0.1665297, -0.4829350, 0.1664852, -0.6029208, 0.6018180
4: -1.9463906, -0.7913225, -1.9463785, -0.7912730, -0.6373682, 0.6362840
5: -0.6560019, 0.0363577, -0.6559807, 0.0363225, -0.6326621, 0.6320616
6: -1.0627794, -0.2341248, -1.0627731, -0.2341335, -0.5143057, 0.5143582
7: -2.2466960, -0.7757933, -2.2466691, -0.7757383, -0.8827365, 0.8827381
8: -1.3526502, -0.6113803, -1.3524728, -0.6113862, -0.3821841, 0.3831649
9: -1.1429060, -0.3270670, -1.1426309, -0.3270670, -0.4180642, 0.4175318

Time for backsubstitution: 6.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3036
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 3037
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2854
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2394
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 3441
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 3429
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 3259
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 3321
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 3423
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2815
type: A, layer: 1, pos: 3320
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 3275
type: A, layer: 1, pos: 2753
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3452
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 3436
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 3232
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3271
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2791
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 3422
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 279
type: A, layer: 1, pos: 3564
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2461
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2489
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3269

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3100

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2432198, upper bound: 0.2435495
time: 19.73 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2433729, upper bound: 0.2435486
time: 79.87 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -3.0342937, -1.6305420, -3.0341008, -1.6237535, -1.0062594, 0.9970663
1: -2.3472958, -1.2734900, -2.3470702, -1.2694025, -0.5945706, 0.5883956
2: -1.0462358, -0.0027303, -1.0469401, -0.0027405, -0.8862628, 0.8848619
3: -0.4829558, 0.1665182, -0.4882703, 0.1665211, -0.6029293, 0.6071167
4: -1.9463882, -0.7912796, -1.9485545, -0.7911907, -0.6374152, 0.6385373
5: -0.6560079, 0.0363331, -0.6614110, 0.0363246, -0.6326522, 0.6374293
6: -1.0627828, -0.2341228, -1.0656236, -0.2341133, -0.5143456, 0.5168286
7: -2.2467017, -0.7758350, -2.2514341, -0.7757781, -0.8827302, 0.8874488
8: -1.3527691, -0.6113939, -1.3526704, -0.6089716, -0.3856151, 0.3832965
9: -1.1427197, -0.3270670, -1.1426501, -0.3267658, -0.4187676, 0.4177220

Time for backsubstitution: 6.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3036
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 3037
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2854
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2394
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 3441
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 3429
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 3259
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 3321
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 3423
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2815
type: A, layer: 1, pos: 3320
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 3275
type: A, layer: 1, pos: 2753
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3452
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 3436
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 3232
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3271
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2791
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 3422
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 279
type: A, layer: 1, pos: 3564
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2461
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2489
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3269

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3100

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2433932, upper bound: 0.2435483
time: 172.25 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2435457, upper bound: 0.2435485
time: 89.67 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 268.57 seconds
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 268.57
Output dim: 3, lower bound: -0.2433676, upper bound: 0.2431943
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 268.57
Output dim: 3, lower bound: -0.2435205, upper bound: 0.2431936
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 268.57
Output dim: 3, lower bound: -0.2432198, upper bound: 0.2435495
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 268.57
Output dim: 3, lower bound: -0.2433729, upper bound: 0.2435486
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 268.57
Output dim: 3, lower bound: -0.2433932, upper bound: 0.2435483
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 268.57
Output dim: 3, lower bound: -0.2435457, upper bound: 0.2435485

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.0320024, -1.6373472, -3.0324590, -1.6291449, -0.9989794, 0.9853821
1: -2.3478758, -1.2736416, -2.3461351, -1.2700546, -0.5947561, 0.5842800
2: -1.0435147, -0.0095137, -1.0458519, -0.0087482, -0.8776655, 0.8797936
3: -0.4804524, 0.1630268, -0.4873558, 0.1635386, -0.5972829, 0.6036792
4: -1.9471885, -0.7915810, -1.9471356, -0.7914497, -0.6312783, 0.6356127
5: -0.6534749, 0.0326676, -0.6603202, 0.0333423, -0.6268739, 0.6330803
6: -1.0615132, -0.2352256, -1.0652905, -0.2343590, -0.5127875, 0.5158449
7: -2.2443759, -0.7768687, -2.2488751, -0.7769279, -0.8764293, 0.8830670
8: -1.3531139, -0.6119877, -1.3519340, -0.6095092, -0.3865934, 0.3795728
9: -1.1426197, -0.3276561, -1.1423712, -0.3271302, -0.4189397, 0.4171613

Time for backsubstitution: 6.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 3037
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2854
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2394
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 413
type: B, layer: 1, pos: 3441
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 2853
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 3429
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 3263
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3321
type: B, layer: 1, pos: 3423
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2815
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 2753
type: B, layer: 1, pos: 3319
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3452
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 3436
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 2400
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3271
type: B, layer: 1, pos: 3232
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2771
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2791
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 3422
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 279
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2461
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3269

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2598

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2432899, upper bound: 0.2431629
time: 9.54 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2434905, upper bound: 0.2431659
time: 8.48 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.0341063, -1.6348536, -3.0334473, -1.6343460, -0.9930990, 0.9917177
1: -2.3476870, -1.2768979, -2.3470950, -1.2764339, -0.5852697, 0.5845367
2: -1.0450047, -0.0027111, -1.0451565, -0.0027177, -0.8848633, 0.8824167
3: -0.4788390, 0.1665219, -0.4792695, 0.1664789, -0.5987402, 0.5980915
4: -1.9441508, -0.7913312, -1.9443715, -0.7912810, -0.6344438, 0.6336972
5: -0.6518055, 0.0363557, -0.6522313, 0.0363208, -0.6283976, 0.6282538
6: -1.0615412, -0.2341465, -1.0616513, -0.2341517, -0.5130060, 0.5131741
7: -2.2422371, -0.7757953, -2.2426658, -0.7757403, -0.8775564, 0.8782380
8: -1.3526437, -0.6139311, -1.3524679, -0.6136354, -0.3796547, 0.3801577
9: -1.1427829, -0.3273806, -1.1425219, -0.3273267, -0.4175540, 0.4169389

Time for backsubstitution: 6.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 3037
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2854
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 413
type: B, layer: 1, pos: 2394
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 3441
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 2853
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 3429
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 3263
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 3321
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3423
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2815
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 2753
type: B, layer: 1, pos: 3319
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3452
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 3436
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 2400
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 3232
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3271
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2771
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2791
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 3422
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 279
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2461
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3269

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2598

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2429916, upper bound: 0.2435162
time: 37.77 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2431893, upper bound: 0.2435170
time: 7.30 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.0359540, -1.6306074, -3.0334725, -1.6306896, -1.0000479, 0.9940252
1: -2.3495436, -1.2737408, -2.3471665, -1.2737739, -0.5907676, 0.5866694
2: -1.0458095, -0.0023897, -1.0457944, -0.0027177, -0.8859369, 0.8837870
3: -0.4826325, 0.1666523, -0.4825299, 0.1664838, -0.6025729, 0.6015475
4: -1.9456105, -0.7911539, -1.9456476, -0.7912751, -0.6358388, 0.6368392
5: -0.6556907, 0.0363906, -0.6555908, 0.0363221, -0.6323243, 0.6317642
6: -1.0625784, -0.2341450, -1.0625563, -0.2341372, -0.5140564, 0.5141407
7: -2.2461379, -0.7754745, -2.2460122, -0.7757388, -0.8815446, 0.8835142
8: -1.3539006, -0.6114509, -1.3524700, -0.6114722, -0.3838611, 0.3815102
9: -1.1433496, -0.3272096, -1.1425951, -0.3272270, -0.4189261, 0.4172195

Time for backsubstitution: 6.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 3037
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2854
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2394
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 413
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 3441
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 2853
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 3429
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 3263
type: B, layer: 1, pos: 3321
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3423
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2815
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 2753
type: B, layer: 1, pos: 3319
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3452
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 3436
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 2400
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3232
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3271
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2771
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2791
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 3422
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 279
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2461
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3269

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2598

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2431422, upper bound: 0.2435195
time: 10.32 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2433421, upper bound: 0.2435151
time: 10.13 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.0342517, -1.6348543, -3.0340657, -1.6275654, -1.0018471, 0.9919205
1: -2.3471758, -1.2768974, -2.3469625, -1.2723460, -0.5910849, 0.5842705
2: -1.0449882, -0.0027303, -1.0458562, -0.0027405, -0.8846763, 0.8834203
3: -0.4788403, 0.1665103, -0.4846021, 0.1665147, -0.5987487, 0.6033888
4: -1.9441490, -0.7912883, -1.9465466, -0.7911978, -0.6344903, 0.6359481
5: -0.6518115, 0.0363311, -0.6576591, 0.0363230, -0.6283880, 0.6336190
6: -1.0615447, -0.2341442, -1.0645000, -0.2341317, -0.5130458, 0.5156450
7: -2.2422428, -0.7758368, -2.2474265, -0.7757798, -0.8775502, 0.8829404
8: -1.3527632, -0.6139446, -1.3526654, -0.6112210, -0.3830856, 0.3802893
9: -1.1425964, -0.3273807, -1.1425416, -0.3270254, -0.4182573, 0.4171293

Time for backsubstitution: 6.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 3037
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2854
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 2394
type: B, layer: 1, pos: 413
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 3441
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 2853
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 3429
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 3263
type: B, layer: 1, pos: 3321
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3423
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2815
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 2753
type: B, layer: 1, pos: 3319
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 3452
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 3436
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 2400
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 3232
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3271
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2771
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2791
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 3422
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 279
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2461
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3269

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2598

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2431647, upper bound: 0.2435192
time: 17.60 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2433628, upper bound: 0.2431611
time: 446.81 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.0360999, -1.6306076, -3.0340903, -1.6239095, -1.0087959, 0.9942278
1: -2.3490326, -1.2737404, -2.3470330, -1.2696856, -0.5965829, 0.5864025
2: -1.0457928, -0.0024088, -1.0464947, -0.0027406, -0.8857496, 0.8847914
3: -0.4826338, 0.1666408, -0.4878649, 0.1665197, -0.6025814, 0.6068462
4: -1.9456090, -0.7911111, -1.9478230, -0.7911925, -0.6358858, 0.6390924
5: -0.6556966, 0.0363661, -0.6610205, 0.0363241, -0.6323146, 0.6371315
6: -1.0625818, -0.2341431, -1.0654064, -0.2341171, -0.5140963, 0.5166111
7: -2.2461433, -0.7755161, -2.2507751, -0.7757783, -0.8815387, 0.8882240
8: -1.3540200, -0.6114647, -1.3526678, -0.6090584, -0.3872921, 0.3816419
9: -1.1431628, -0.3272095, -1.1426144, -0.3269258, -0.4196293, 0.4174097

Time for backsubstitution: 6.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 3037
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2854
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2394
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 413
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 3441
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 2853
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 3429
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 3263
type: B, layer: 1, pos: 3321
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3423
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2815
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 2753
type: B, layer: 1, pos: 3319
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 3452
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 3436
type: B, layer: 1, pos: 2400
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3232
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3271
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2771
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2791
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 3422
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 279
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 751
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 239
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 2354
type: B, layer: 1, pos: 2414
type: B, layer: 1, pos: 2461
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2489
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 3044
type: B, layer: 1, pos: 3074
type: B, layer: 1, pos: 3104
type: B, layer: 1, pos: 3269

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2598

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2433181, upper bound: 0.2435199
time: 8.49 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2435152, upper bound: 0.2435194
time: 13.32 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 28.41 seconds
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 28.41
Output dim: 3, lower bound: -0.2432899, upper bound: 0.2431629
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.41
Output dim: 3, lower bound: -0.2434905, upper bound: 0.2431659
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.41
Output dim: 3, lower bound: -0.2429916, upper bound: 0.2435162
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.41
Output dim: 3, lower bound: -0.2431893, upper bound: 0.2435170
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.41
Output dim: 3, lower bound: -0.2431422, upper bound: 0.2435195
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.41
Output dim: 3, lower bound: -0.2433421, upper bound: 0.2435151
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.41
Output dim: 3, lower bound: -0.2431647, upper bound: 0.2435192
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 28.41
Output dim: 3, lower bound: -0.2433628, upper bound: 0.2431611
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.41
Output dim: 3, lower bound: -0.2433181, upper bound: 0.2435199
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.41
Output dim: 3, lower bound: -0.2435152, upper bound: 0.2435194

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.0308762, -1.6373467, -3.0311522, -1.6291459, -0.9983172, 0.9785458
1: -2.3463392, -1.2736430, -2.3443086, -1.2700565, -0.5934453, 0.5784912
2: -1.0434935, -0.0095892, -1.0458257, -0.0088402, -0.8773670, 0.8797973
3: -0.4804223, 0.1629351, -0.4873178, 0.1634266, -0.5971180, 0.6035539
4: -1.9471629, -0.7916443, -1.9471051, -0.7915282, -0.6310651, 0.6355866
5: -0.6534452, 0.0326149, -0.6602829, 0.0332773, -0.6267738, 0.6330023
6: -1.0615091, -0.2353680, -1.0652857, -0.2345347, -0.5118555, 0.5157399
7: -2.2443252, -0.7769495, -2.2488127, -0.7770271, -0.8756385, 0.8829472
8: -1.3528647, -0.6119964, -1.3516769, -0.6095197, -0.3864036, 0.3762785
9: -1.1417674, -0.3276794, -1.1413448, -0.3271591, -0.4180967, 0.4161143

Time for backsubstitution: 6.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3036
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 3037
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2854
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2394
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 3441
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 3429
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 3259
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 3321
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 3423
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2815
type: A, layer: 1, pos: 3320
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 3275
type: A, layer: 1, pos: 2753
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3452
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 3436
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 3271
type: A, layer: 1, pos: 3232
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2791
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 3422
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 279
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 3564
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2461
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2489
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3269

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2630

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2434432, upper bound: 0.2429799
time: 57.92 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2434442, upper bound: 0.2431151
time: 130.05 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3.0295830, -1.6348557, -3.0279074, -1.6363037, -0.9861488, 0.9863088
1: -2.3417103, -1.2769046, -2.3398030, -1.2778761, -0.5772086, 0.5770655
2: -1.0449259, -0.0029045, -1.0448261, -0.0029546, -0.8843949, 0.8814532
3: -0.4786988, 0.1662383, -0.4771563, 0.1661264, -0.5982465, 0.5956480
4: -1.9440668, -0.7915225, -1.9438894, -0.7915156, -0.6340582, 0.6328604
5: -0.6516489, 0.0362096, -0.6501859, 0.0361378, -0.6280214, 0.6259145
6: -1.0614945, -0.2345127, -1.0602772, -0.2345985, -0.5124428, 0.5110589
7: -2.2420306, -0.7762026, -2.2409165, -0.7762315, -0.8769115, 0.8759845
8: -1.3505721, -0.6139554, -1.3499899, -0.6148601, -0.3764546, 0.3777301
9: -1.1396463, -0.3274358, -1.1387529, -0.3273967, -0.4139116, 0.4126791

Time for backsubstitution: 6.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3036
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 3037
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2854
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2394
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 3441
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 3429
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 3259
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 3321
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 3423
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2815
type: A, layer: 1, pos: 3320
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 3275
type: A, layer: 1, pos: 2753
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3452
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 3436
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 3232
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3271
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2791
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 3422
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 279
type: A, layer: 1, pos: 3564
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2461
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2489
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3269

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2630

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2429462, upper bound: 0.2433354
time: 37.20 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2429466, upper bound: 0.2434685
time: 832.74 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.0329785, -1.6348541, -3.0321474, -1.6343465, -0.9924372, 0.9848794
1: -2.3461499, -1.2768991, -2.3452673, -1.2764359, -0.5839593, 0.5787498
2: -1.0449833, -0.0027868, -1.0451303, -0.0028098, -0.8845648, 0.8824205
3: -0.4788082, 0.1664300, -0.4792316, 0.1663666, -0.5985742, 0.5979663
4: -1.9441266, -0.7913953, -1.9443412, -0.7913593, -0.6342306, 0.6336704
5: -0.6517750, 0.0363026, -0.6521938, 0.0362555, -0.6282966, 0.6281756
6: -1.0615368, -0.2342888, -1.0616462, -0.2343275, -0.5120738, 0.5130692
7: -2.2421861, -0.7758756, -2.2426033, -0.7758394, -0.8767638, 0.8781176
8: -1.3523948, -0.6139396, -1.3522109, -0.6136458, -0.3794649, 0.3768644
9: -1.1419313, -0.3274041, -1.1414956, -0.3273556, -0.4167117, 0.4158927

Time for backsubstitution: 6.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3036
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 3037
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2854
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2394
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 3441
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 3429
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 3259
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 3321
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 3423
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2815
type: A, layer: 1, pos: 3320
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 3275
type: A, layer: 1, pos: 2753
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3452
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 3436
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 3232
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3271
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2791
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 3422
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 279
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 3564
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2461
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2489
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3269

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2630

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2431427, upper bound: 0.2433387
time: 7.98 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2431467, upper bound: 0.2434712
time: 119.22 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3.0314317, -1.6306098, -3.0279317, -1.6326482, -0.9930972, 0.9886163
1: -2.3435669, -1.2737474, -2.3398743, -1.2752155, -0.5827076, 0.5791981
2: -1.0457307, -0.0025831, -1.0454636, -0.0029547, -0.8854683, 0.8828233
3: -0.4824925, 0.1663687, -0.4804162, 0.1661314, -0.6020800, 0.5991049
4: -1.9455270, -0.7913454, -1.9451632, -0.7915103, -0.6354543, 0.6360023
5: -0.6555344, 0.0362445, -0.6535460, 0.0361389, -0.6319481, 0.6294255
6: -1.0625315, -0.2345114, -1.0611818, -0.2345840, -0.5134934, 0.5120251
7: -2.2459309, -0.7758820, -2.2442632, -0.7762300, -0.8809012, 0.8812592
8: -1.3518286, -0.6114755, -1.3499925, -0.6126976, -0.3806612, 0.3790827
9: -1.1402123, -0.3272645, -1.1388257, -0.3272971, -0.4152835, 0.4129593

Time for backsubstitution: 6.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3036
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 3037
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2854
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2212
type: A, layer: 1, pos: 2394
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 3441
type: A, layer: 1, pos: 2600
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 3429
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 3259
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 3321
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 3423
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2815
type: A, layer: 1, pos: 3320
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 3275
type: A, layer: 1, pos: 2753
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 3452
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 3436
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 3232
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3271
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2791
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 3422
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 279
type: A, layer: 1, pos: 3564
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 751
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 239
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 2354
type: A, layer: 1, pos: 2414
type: A, layer: 1, pos: 2461
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2489
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 3044
type: A, layer: 1, pos: 3074
type: A, layer: 1, pos: 3104
type: A, layer: 1, pos: 3269

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2630

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2430975, upper bound: 0.2429805
time: 539.72 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2430985, upper bound: 0.2434707
time: 185.71 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 98.89 + 3748.13 = 3847.02 seconds
