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
execution time: IAR + RelationalAnalysis = 7.31 + 91.21 = 98.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.2436462, upper bound: 0.2436487

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2600
type: B, layer: 1, pos: 2600
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 413
type: A, layer: 1, pos: 413
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2630
type: B, layer: 1, pos: 2630
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 3100
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 3036
type: B, layer: 1, pos: 3036
type: A, layer: 1, pos: 2402
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 3032
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2584
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 2185
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 295
type: B, layer: 1, pos: 295
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 2421
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2587
type: B, layer: 1, pos: 2587
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 3022
type: A, layer: 1, pos: 3022
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 3037
type: B, layer: 1, pos: 3037
type: B, layer: 1, pos: 2642
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 3441
type: B, layer: 1, pos: 3097
type: A, layer: 1, pos: 3097
type: B, layer: 1, pos: 423
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 2422
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2854
type: A, layer: 1, pos: 2854
type: A, layer: 1, pos: 2327
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 2992
type: A, layer: 1, pos: 2992
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2212
type: B, layer: 1, pos: 2212
type: A, layer: 1, pos: 2991
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2511
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 3423
type: B, layer: 1, pos: 3423
type: B, layer: 1, pos: 2616
type: A, layer: 1, pos: 2616
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 3429
type: B, layer: 1, pos: 3429
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 3008
type: A, layer: 1, pos: 2417
type: B, layer: 1, pos: 2417
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 426
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2317
type: A, layer: 1, pos: 2317
type: B, layer: 1, pos: 2853
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2522
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 2526
type: B, layer: 1, pos: 2526
type: A, layer: 1, pos: 3436
type: B, layer: 1, pos: 3436
type: B, layer: 1, pos: 2380
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 3070
type: B, layer: 1, pos: 3070
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 3452
type: B, layer: 1, pos: 3452
type: A, layer: 1, pos: 2060
type: B, layer: 1, pos: 2060
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 3259
type: B, layer: 1, pos: 3259
type: A, layer: 1, pos: 2873
type: B, layer: 1, pos: 2873
type: A, layer: 1, pos: 2863
type: B, layer: 1, pos: 2863
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 279
type: B, layer: 1, pos: 279
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 2079
type: B, layer: 1, pos: 2079
type: A, layer: 1, pos: 2627
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 3275
type: B, layer: 1, pos: 3275
type: A, layer: 1, pos: 3320
type: B, layer: 1, pos: 3320
type: A, layer: 1, pos: 500
type: B, layer: 1, pos: 500
type: A, layer: 1, pos: 427
type: B, layer: 1, pos: 427
type: A, layer: 1, pos: 3321
type: B, layer: 1, pos: 3321
type: A, layer: 1, pos: 2753
type: B, layer: 1, pos: 2753
type: B, layer: 1, pos: 3502
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 3187
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 2423
type: B, layer: 1, pos: 2423
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 3319
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 2400
type: B, layer: 1, pos: 2400
type: A, layer: 1, pos: 3263
type: B, layer: 1, pos: 3263
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 2547
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2160
type: A, layer: 1, pos: 2097
type: B, layer: 1, pos: 2097
type: A, layer: 1, pos: 3232
type: B, layer: 1, pos: 3232
type: A, layer: 1, pos: 3090
type: B, layer: 1, pos: 3090
type: A, layer: 1, pos: 2250
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 2815
type: B, layer: 1, pos: 2815
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2791
type: B, layer: 1, pos: 2791
type: A, layer: 1, pos: 3271
type: B, layer: 1, pos: 3271
type: A, layer: 1, pos: 2394
type: B, layer: 1, pos: 2394
type: A, layer: 1, pos: 2866
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 2139
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 2328
type: B, layer: 1, pos: 2328
type: A, layer: 1, pos: 3453
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 2577
type: A, layer: 1, pos: 2577
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 2668
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 2095
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3422
type: B, layer: 1, pos: 3422
type: A, layer: 1, pos: 3056
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 2589
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2580
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 3339
type: B, layer: 1, pos: 3339
type: A, layer: 1, pos: 506
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 2352
type: B, layer: 1, pos: 2352
type: A, layer: 1, pos: 2771
type: B, layer: 1, pos: 2771
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 3565
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 3057
type: B, layer: 1, pos: 3057
type: A, layer: 1, pos: 3457
type: B, layer: 1, pos: 3457
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 3286
type: A, layer: 1, pos: 3286
type: B, layer: 1, pos: 2550
type: A, layer: 1, pos: 2550
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3564
type: A, layer: 1, pos: 3267
type: B, layer: 1, pos: 3267
type: A, layer: 1, pos: 3028
type: B, layer: 1, pos: 3028
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2326
type: B, layer: 1, pos: 2326
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
type: A, layer: 1, pos: 2600

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2435746, upper bound: 0.2434025
time: 144.39 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2435761, upper bound: 0.2435788
time: 71.38 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 215.85 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 215.85
Output dim: 3, lower bound: -0.2435746, upper bound: 0.2434025
NS_A2, status: Status.UNKNOWN, split count: 1, time: 215.85
Output dim: 3, lower bound: -0.2435761, upper bound: 0.2435788

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3.0334940, -1.6305058, -3.0341635, -1.6305056, -0.9969103, 0.9975049
1: -2.3480752, -1.2734900, -2.3489203, -1.2734885, -0.5894952, 0.5903269
2: -1.0462484, -0.0027027, -1.0462630, -0.0026915, -0.8864877, 0.8864787
3: -0.4829383, 0.1664909, -0.4829586, 0.1665370, -0.6029612, 0.6029299
4: -1.9463836, -0.7905626, -1.9463959, -0.7904251, -0.6368154, 0.6366938
5: -0.6559840, 0.0363251, -0.6560059, 0.0363609, -0.6326852, 0.6326689
6: -1.0627737, -0.2341326, -1.0627799, -0.2341240, -0.5147122, 0.5147097
7: -2.2466788, -0.7753797, -2.2467086, -0.7753412, -0.8827531, 0.8827450
8: -1.3526214, -0.6113840, -1.3528444, -0.6113787, -0.3831751, 0.3833838
9: -1.1430211, -0.3270662, -1.1434064, -0.3270662, -0.4180715, 0.4184920

Time for backsubstitution: 5.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2598
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 413
type: A, layer: 1, pos: 413
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2630
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3100
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 3036
type: B, layer: 1, pos: 3036
type: A, layer: 1, pos: 2402
type: B, layer: 1, pos: 2402
type: A, layer: 1, pos: 3032
type: B, layer: 1, pos: 3032
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2584
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 2185
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 295
type: B, layer: 1, pos: 295
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2421
type: B, layer: 1, pos: 2587
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 3022
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 3037
type: A, layer: 1, pos: 3037
type: B, layer: 1, pos: 2642
type: A, layer: 1, pos: 2642
type: B, layer: 1, pos: 3441
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 3097
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 423
type: B, layer: 1, pos: 423
type: A, layer: 1, pos: 2422
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2854
type: A, layer: 1, pos: 2854
type: A, layer: 1, pos: 2327
type: B, layer: 1, pos: 2327
type: A, layer: 1, pos: 2992
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2212
type: B, layer: 1, pos: 2212
type: A, layer: 1, pos: 2991
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2511
type: A, layer: 1, pos: 2511
type: B, layer: 1, pos: 3423
type: A, layer: 1, pos: 3423
type: B, layer: 1, pos: 2616
type: A, layer: 1, pos: 2616
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 3429
type: A, layer: 1, pos: 3429
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 3008
type: A, layer: 1, pos: 2417
type: B, layer: 1, pos: 2417
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 426
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 2317
type: A, layer: 1, pos: 2317
type: B, layer: 1, pos: 2853
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2522
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 2526
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 3436
type: A, layer: 1, pos: 3436
type: A, layer: 1, pos: 2380
type: B, layer: 1, pos: 2380
type: A, layer: 1, pos: 3070
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 3452
type: A, layer: 1, pos: 3452
type: B, layer: 1, pos: 2060
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 3259
type: A, layer: 1, pos: 3259
type: B, layer: 1, pos: 2873
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 2863
type: B, layer: 1, pos: 2863
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 279
type: B, layer: 1, pos: 279
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 2079
type: B, layer: 1, pos: 2079
type: A, layer: 1, pos: 2627
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 3275
type: B, layer: 1, pos: 3275
type: A, layer: 1, pos: 3287
type: B, layer: 1, pos: 3320
type: A, layer: 1, pos: 3320
type: B, layer: 1, pos: 500
type: A, layer: 1, pos: 500
type: B, layer: 1, pos: 427
type: A, layer: 1, pos: 427
type: B, layer: 1, pos: 3321
type: A, layer: 1, pos: 3321
type: B, layer: 1, pos: 2753
type: A, layer: 1, pos: 2753
type: B, layer: 1, pos: 3502
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 2423
type: B, layer: 1, pos: 3187
type: A, layer: 1, pos: 3187
type: B, layer: 1, pos: 2423
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 3319
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 2400
type: B, layer: 1, pos: 2400
type: B, layer: 1, pos: 3263
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2547
type: A, layer: 1, pos: 2547
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 2097
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 3232
type: A, layer: 1, pos: 3232
type: B, layer: 1, pos: 3090
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2250
type: B, layer: 1, pos: 2250
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 2815
type: B, layer: 1, pos: 2815
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2791
type: A, layer: 1, pos: 2791
type: B, layer: 1, pos: 3271
type: A, layer: 1, pos: 3271
type: A, layer: 1, pos: 2394
type: B, layer: 1, pos: 2866
type: A, layer: 1, pos: 2866
type: B, layer: 1, pos: 2394
type: A, layer: 1, pos: 2328
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 2139
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 2577
type: B, layer: 1, pos: 3453
type: A, layer: 1, pos: 3453
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 2577
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 2668
type: B, layer: 1, pos: 2668
type: A, layer: 1, pos: 2095
type: B, layer: 1, pos: 2095
type: A, layer: 1, pos: 3056
type: B, layer: 1, pos: 3422
type: A, layer: 1, pos: 3422
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 3339
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 506
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 2352
type: B, layer: 1, pos: 2352
type: A, layer: 1, pos: 2771
type: B, layer: 1, pos: 2771
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 3057
type: A, layer: 1, pos: 3565
type: B, layer: 1, pos: 3565
type: A, layer: 1, pos: 3057
type: B, layer: 1, pos: 3457
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 3286
type: B, layer: 1, pos: 3286
type: A, layer: 1, pos: 2550
type: B, layer: 1, pos: 2550
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 3267
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 3028
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2326
type: A, layer: 1, pos: 2326
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

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2433467, upper bound: 0.2433701
time: 195.53 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2435445, upper bound: 0.2433745
time: 14.53 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -3.0341125, -1.6237259, -3.0343091, -1.6305058, -0.9971131, 1.0062534
1: -2.3479419, -1.2694014, -2.3484087, -1.2734883, -0.5892298, 0.5961410
2: -1.0469487, -0.0027256, -1.0462463, -0.0027106, -0.8874919, 0.8862916
3: -0.4882737, 0.1665268, -0.4829599, 0.1665255, -0.6082596, 0.6029387
4: -1.9485588, -0.7904798, -1.9463940, -0.7903823, -0.6390687, 0.6367403
5: -0.6614143, 0.0363272, -0.6560121, 0.0363364, -0.6380528, 0.6326591
6: -1.0656240, -0.2341126, -1.0627835, -0.2341220, -0.5171846, 0.5147497
7: -2.2514434, -0.7754190, -2.2467144, -0.7753825, -0.8874638, 0.8827388
8: -1.3528187, -0.6089700, -1.3529633, -0.6113923, -0.3833067, 0.3868150
9: -1.1430398, -0.3267652, -1.1432202, -0.3270661, -0.4182613, 0.4191952

Time for backsubstitution: 5.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2598
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 413
type: A, layer: 1, pos: 413
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2630
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3100
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 3036
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 3032
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 3032
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2584
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 2185
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 295
type: B, layer: 1, pos: 295
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2421
type: B, layer: 1, pos: 2587
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 3022
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 3037
type: A, layer: 1, pos: 3037
type: B, layer: 1, pos: 2642
type: A, layer: 1, pos: 2642
type: B, layer: 1, pos: 3441
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 3097
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 423
type: B, layer: 1, pos: 423
type: A, layer: 1, pos: 2422
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2854
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 2854
type: B, layer: 1, pos: 2327
type: A, layer: 1, pos: 2992
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2212
type: B, layer: 1, pos: 2212
type: A, layer: 1, pos: 2991
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 2616
type: A, layer: 1, pos: 2511
type: B, layer: 1, pos: 2991
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2417
type: B, layer: 1, pos: 3423
type: A, layer: 1, pos: 3423
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 3429
type: A, layer: 1, pos: 3429
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 3008
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2616
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 426
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2853
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2317
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2522
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2526
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 3436
type: A, layer: 1, pos: 3436
type: A, layer: 1, pos: 2380
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 3452
type: A, layer: 1, pos: 3452
type: B, layer: 1, pos: 2060
type: A, layer: 1, pos: 2060
type: B, layer: 1, pos: 2600
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 3259
type: A, layer: 1, pos: 3259
type: B, layer: 1, pos: 2873
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 2863
type: B, layer: 1, pos: 2863
type: A, layer: 1, pos: 279
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 279
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2079
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 3275
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 427
type: A, layer: 1, pos: 3287
type: B, layer: 1, pos: 3320
type: A, layer: 1, pos: 3320
type: B, layer: 1, pos: 500
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 427
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 2753
type: B, layer: 1, pos: 3321
type: A, layer: 1, pos: 3321
type: B, layer: 1, pos: 3502
type: A, layer: 1, pos: 2753
type: B, layer: 1, pos: 3187
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 3319
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 2400
type: B, layer: 1, pos: 2400
type: B, layer: 1, pos: 3263
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2160
type: A, layer: 1, pos: 2394
type: A, layer: 1, pos: 2097
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 3232
type: A, layer: 1, pos: 3232
type: B, layer: 1, pos: 3090
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2250
type: B, layer: 1, pos: 2250
type: A, layer: 1, pos: 2815
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 2815
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3056
type: B, layer: 1, pos: 2791
type: A, layer: 1, pos: 2791
type: B, layer: 1, pos: 3271
type: A, layer: 1, pos: 3271
type: B, layer: 1, pos: 2866
type: A, layer: 1, pos: 2866
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 2139
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 2095
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 2668
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3422
type: A, layer: 1, pos: 3422
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3339
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 2577
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 506
type: A, layer: 1, pos: 506
type: B, layer: 1, pos: 2394
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 2352
type: B, layer: 1, pos: 3057
type: A, layer: 1, pos: 2771
type: B, layer: 1, pos: 2352
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 2771
type: A, layer: 1, pos: 2589
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 3565
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 3457
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 3028
type: A, layer: 1, pos: 3286
type: B, layer: 1, pos: 3286
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 2550
type: B, layer: 1, pos: 2550
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 3267
type: A, layer: 1, pos: 3267
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2926
type: A, layer: 1, pos: 3028
type: B, layer: 1, pos: 2326
type: A, layer: 1, pos: 2326
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
type: B, layer: 1, pos: 2328

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2598

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2433467, upper bound: 0.2435451
time: 8.44 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2435446, upper bound: 0.2435444
time: 87.63 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 102.02 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 102.02
Output dim: 3, lower bound: -0.2433467, upper bound: 0.2433701
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 102.02
Output dim: 3, lower bound: -0.2435445, upper bound: 0.2433745
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 102.02
Output dim: 3, lower bound: -0.2433467, upper bound: 0.2435451
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 102.02
Output dim: 3, lower bound: -0.2435446, upper bound: 0.2435444

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -3.0323670, -1.6305065, -3.0328631, -1.6305060, -0.9962485, 0.9906665
1: -2.3465376, -1.2734911, -2.3470917, -1.2734909, -0.5881847, 0.5845404
2: -1.0462272, -0.0027783, -1.0462370, -0.0027834, -0.8861890, 0.8864826
3: -0.4829075, 0.1663991, -0.4829209, 0.1664249, -0.6027951, 0.6028047
4: -1.9463586, -0.7906263, -1.9463655, -0.7905037, -0.6366022, 0.6366675
5: -0.6559535, 0.0362719, -0.6559685, 0.0362955, -0.6325841, 0.6325907
6: -1.0627691, -0.2342752, -1.0627747, -0.2342998, -0.5137800, 0.5146047
7: -2.2466278, -0.7754602, -2.2466455, -0.7754401, -0.8819613, 0.8826251
8: -1.3523722, -0.6113929, -1.3525873, -0.6113889, -0.3829854, 0.3800905
9: -1.1421688, -0.3270896, -1.1423798, -0.3270951, -0.4172287, 0.4174453

Time for backsubstitution: 5.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 413
type: B, layer: 1, pos: 413
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2630
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 3100
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 3036
type: B, layer: 1, pos: 2402
type: A, layer: 1, pos: 2402
type: B, layer: 1, pos: 3032
type: A, layer: 1, pos: 3032
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2584
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 2185
type: B, layer: 1, pos: 295
type: A, layer: 1, pos: 295
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2421
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2587
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3022
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 3037
type: B, layer: 1, pos: 3037
type: A, layer: 1, pos: 2642
type: B, layer: 1, pos: 2642
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 3441
type: A, layer: 1, pos: 3097
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 423
type: A, layer: 1, pos: 423
type: B, layer: 1, pos: 2422
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2854
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 2854
type: A, layer: 1, pos: 2327
type: B, layer: 1, pos: 2992
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2441
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2212
type: A, layer: 1, pos: 2212
type: B, layer: 1, pos: 2991
type: A, layer: 1, pos: 2511
type: B, layer: 1, pos: 2511
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3423
type: B, layer: 1, pos: 3423
type: B, layer: 1, pos: 2417
type: A, layer: 1, pos: 3429
type: B, layer: 1, pos: 3429
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 3008
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 426
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 2853
type: B, layer: 1, pos: 2853
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2522
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 3070
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 3436
type: B, layer: 1, pos: 3436
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 2600
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 3452
type: B, layer: 1, pos: 3452
type: A, layer: 1, pos: 2060
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 3259
type: B, layer: 1, pos: 3259
type: A, layer: 1, pos: 2873
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 2863
type: A, layer: 1, pos: 2863
type: B, layer: 1, pos: 279
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 279
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 2079
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 3287
type: B, layer: 1, pos: 3275
type: A, layer: 1, pos: 3275
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 3320
type: B, layer: 1, pos: 3320
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 500
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 427
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 3321
type: A, layer: 1, pos: 2753
type: B, layer: 1, pos: 3321
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 2753
type: A, layer: 1, pos: 3187
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 3319
type: B, layer: 1, pos: 3319
type: B, layer: 1, pos: 2400
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 3263
type: B, layer: 1, pos: 3263
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2160
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2097
type: A, layer: 1, pos: 2097
type: B, layer: 1, pos: 2394
type: A, layer: 1, pos: 3232
type: B, layer: 1, pos: 3232
type: A, layer: 1, pos: 3090
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2250
type: A, layer: 1, pos: 2250
type: B, layer: 1, pos: 2815
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 2815
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2423
type: B, layer: 1, pos: 2577
type: A, layer: 1, pos: 2791
type: B, layer: 1, pos: 2791
type: A, layer: 1, pos: 3271
type: B, layer: 1, pos: 3271
type: B, layer: 1, pos: 3056
type: A, layer: 1, pos: 2866
type: B, layer: 1, pos: 2866
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 2139
type: A, layer: 1, pos: 3453
type: B, layer: 1, pos: 3453
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 2668
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3422
type: B, layer: 1, pos: 3422
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2580
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3339
type: A, layer: 1, pos: 2394
type: A, layer: 1, pos: 3339
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 506
type: B, layer: 1, pos: 506
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2352
type: B, layer: 1, pos: 2589
type: A, layer: 1, pos: 2352
type: B, layer: 1, pos: 2771
type: A, layer: 1, pos: 3057
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3565
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 3286
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 3028
type: B, layer: 1, pos: 2550
type: A, layer: 1, pos: 2550
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3564
type: A, layer: 1, pos: 3267
type: B, layer: 1, pos: 3267
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 3028
type: A, layer: 1, pos: 2326
type: B, layer: 1, pos: 2326
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
type: A, layer: 1, pos: 413

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2435143, upper bound: 0.2430129
time: 83.16 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2435385, upper bound: 0.2433678
time: 218.84 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -3.0295892, -1.6237278, -3.0287690, -1.6324639, -0.9901624, 1.0008445
1: -2.3419662, -1.2694086, -2.3411164, -1.2749306, -0.5811695, 0.5886692
2: -1.0468706, -0.0029189, -1.0459157, -0.0029475, -0.8870239, 0.8853279
3: -0.4881326, 0.1662432, -0.4808463, 0.1661730, -0.6077657, 0.6004961
4: -1.9484742, -0.7906713, -1.9459097, -0.7906173, -0.6386857, 0.6359038
5: -0.6612576, 0.0361810, -0.6539675, 0.0361532, -0.6376765, 0.6303203
6: -1.0655789, -0.2344792, -1.0614088, -0.2345686, -0.5166206, 0.5126342
7: -2.2512374, -0.7758268, -2.2449660, -0.7758741, -0.8868222, 0.8804843
8: -1.3507469, -0.6089945, -1.3504857, -0.6126169, -0.3801071, 0.3843873
9: -1.1399010, -0.3268203, -1.1394489, -0.3271360, -0.4146186, 0.4149334

Time for backsubstitution: 5.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 413
type: A, layer: 1, pos: 413
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2630
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3100
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 3036
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2402
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 3032
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2584
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 2185
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 295
type: B, layer: 1, pos: 295
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2421
type: B, layer: 1, pos: 2587
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2587
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 3022
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3037
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3037
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 3441
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 3097
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 423
type: B, layer: 1, pos: 423
type: A, layer: 1, pos: 2422
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2854
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 2854
type: B, layer: 1, pos: 2327
type: A, layer: 1, pos: 2992
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2212
type: B, layer: 1, pos: 2212
type: A, layer: 1, pos: 2991
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2511
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2511
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2991
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 3423
type: A, layer: 1, pos: 3423
type: B, layer: 1, pos: 3429
type: A, layer: 1, pos: 3429
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 3008
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 426
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 2853
type: A, layer: 1, pos: 2853
type: B, layer: 1, pos: 2417
type: A, layer: 1, pos: 2317
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2522
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2526
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 2526
type: A, layer: 1, pos: 2380
type: B, layer: 1, pos: 3436
type: A, layer: 1, pos: 3436
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 3452
type: A, layer: 1, pos: 3452
type: B, layer: 1, pos: 2060
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 3259
type: A, layer: 1, pos: 3259
type: B, layer: 1, pos: 2873
type: A, layer: 1, pos: 2873
type: B, layer: 1, pos: 3070
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 279
type: B, layer: 1, pos: 2863
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 279
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 2079
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 427
type: A, layer: 1, pos: 3275
type: B, layer: 1, pos: 3275
type: A, layer: 1, pos: 3287
type: B, layer: 1, pos: 3320
type: A, layer: 1, pos: 3320
type: B, layer: 1, pos: 500
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 2753
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 3321
type: A, layer: 1, pos: 3321
type: B, layer: 1, pos: 3502
type: A, layer: 1, pos: 2753
type: B, layer: 1, pos: 3187
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 3319
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 2400
type: B, layer: 1, pos: 2400
type: B, layer: 1, pos: 3263
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2394
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2160
type: A, layer: 1, pos: 2097
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 3232
type: A, layer: 1, pos: 3232
type: B, layer: 1, pos: 3090
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2815
type: A, layer: 1, pos: 2250
type: B, layer: 1, pos: 2250
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3056
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2815
type: B, layer: 1, pos: 2791
type: A, layer: 1, pos: 2791
type: B, layer: 1, pos: 3271
type: A, layer: 1, pos: 3271
type: B, layer: 1, pos: 2866
type: A, layer: 1, pos: 2866
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 2589
type: A, layer: 1, pos: 2139
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3422
type: A, layer: 1, pos: 3422
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3339
type: B, layer: 1, pos: 3339
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 506
type: A, layer: 1, pos: 506
type: B, layer: 1, pos: 2577
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2423
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 2352
type: B, layer: 1, pos: 3057
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 2352
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 2771
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 2589
type: B, layer: 1, pos: 2394
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 3286
type: B, layer: 1, pos: 3286
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 2550
type: B, layer: 1, pos: 2550
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 3267
type: A, layer: 1, pos: 3267
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 2326
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 3028
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
type: B, layer: 1, pos: 2328

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 413

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2429892, upper bound: 0.2435138
time: 27.19 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2433401, upper bound: 0.2435379
time: 39.78 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -3.0329845, -1.6237259, -3.0330088, -1.6305063, -0.9964513, 0.9994149
1: -2.3464050, -1.2694030, -2.3465812, -1.2734907, -0.5879186, 0.5903533
2: -1.0469276, -0.0028011, -1.0462205, -0.0028027, -0.8871934, 0.8862956
3: -0.4882427, 0.1664349, -0.4829220, 0.1664134, -0.6080939, 0.6028132
4: -1.9485344, -0.7905436, -1.9463633, -0.7904605, -0.6388557, 0.6367143
5: -0.6613836, 0.0362741, -0.6559747, 0.0362710, -0.6379519, 0.6325809
6: -1.0656195, -0.2342552, -1.0627780, -0.2342976, -0.5162523, 0.5146447
7: -2.2513924, -0.7754999, -2.2466514, -0.7754818, -0.8866725, 0.8826189
8: -1.3525696, -0.6089786, -1.3527067, -0.6114025, -0.3831171, 0.3835215
9: -1.1421881, -0.3267885, -1.1421932, -0.3270950, -0.4174188, 0.4181483

Time for backsubstitution: 6.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 413
type: A, layer: 1, pos: 413
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2630
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3100
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 3036
type: B, layer: 1, pos: 3036
type: A, layer: 1, pos: 2402
type: B, layer: 1, pos: 2402
type: A, layer: 1, pos: 3032
type: B, layer: 1, pos: 3032
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2584
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 2185
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 295
type: B, layer: 1, pos: 295
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2421
type: B, layer: 1, pos: 2587
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 3022
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 3037
type: A, layer: 1, pos: 3037
type: B, layer: 1, pos: 2642
type: A, layer: 1, pos: 2642
type: B, layer: 1, pos: 3441
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 3097
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 423
type: B, layer: 1, pos: 423
type: A, layer: 1, pos: 2422
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2854
type: A, layer: 1, pos: 2854
type: A, layer: 1, pos: 2327
type: B, layer: 1, pos: 2327
type: A, layer: 1, pos: 2992
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2212
type: B, layer: 1, pos: 2212
type: A, layer: 1, pos: 2991
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2511
type: A, layer: 1, pos: 2511
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 3423
type: A, layer: 1, pos: 3423
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2616
type: B, layer: 1, pos: 3429
type: A, layer: 1, pos: 3429
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 2417
type: B, layer: 1, pos: 2417
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 426
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2317
type: A, layer: 1, pos: 2317
type: B, layer: 1, pos: 2853
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 2522
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 2526
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 3436
type: A, layer: 1, pos: 3436
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 3070
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 3452
type: A, layer: 1, pos: 3452
type: B, layer: 1, pos: 2060
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 3259
type: A, layer: 1, pos: 3259
type: B, layer: 1, pos: 2873
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 2863
type: B, layer: 1, pos: 2863
type: A, layer: 1, pos: 279
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 279
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 2079
type: B, layer: 1, pos: 2079
type: A, layer: 1, pos: 2627
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 3275
type: B, layer: 1, pos: 3275
type: A, layer: 1, pos: 3287
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 3320
type: A, layer: 1, pos: 3320
type: B, layer: 1, pos: 500
type: A, layer: 1, pos: 500
type: B, layer: 1, pos: 427
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2423
type: B, layer: 1, pos: 3321
type: A, layer: 1, pos: 3321
type: B, layer: 1, pos: 2753
type: A, layer: 1, pos: 2753
type: B, layer: 1, pos: 3502
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 3187
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 3319
type: A, layer: 1, pos: 3319
type: B, layer: 1, pos: 2423
type: A, layer: 1, pos: 2400
type: B, layer: 1, pos: 2400
type: B, layer: 1, pos: 3263
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 2547
type: A, layer: 1, pos: 2547
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2160
type: A, layer: 1, pos: 2097
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 3232
type: A, layer: 1, pos: 3232
type: B, layer: 1, pos: 3090
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2250
type: B, layer: 1, pos: 2250
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 2815
type: B, layer: 1, pos: 2815
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2394
type: A, layer: 1, pos: 2791
type: B, layer: 1, pos: 2791
type: A, layer: 1, pos: 3271
type: B, layer: 1, pos: 3271
type: A, layer: 1, pos: 2328
type: B, layer: 1, pos: 2866
type: A, layer: 1, pos: 2866
type: B, layer: 1, pos: 2394
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 2577
type: B, layer: 1, pos: 2139
type: A, layer: 1, pos: 2139
type: B, layer: 1, pos: 3453
type: A, layer: 1, pos: 3453
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2577
type: A, layer: 1, pos: 2668
type: B, layer: 1, pos: 2668
type: A, layer: 1, pos: 2095
type: B, layer: 1, pos: 2095
type: A, layer: 1, pos: 3056
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 3422
type: A, layer: 1, pos: 3422
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3339
type: B, layer: 1, pos: 3339
type: A, layer: 1, pos: 2589
type: B, layer: 1, pos: 506
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 2352
type: B, layer: 1, pos: 2352
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2771
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 3565
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 3057
type: B, layer: 1, pos: 3457
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 3286
type: B, layer: 1, pos: 3286
type: A, layer: 1, pos: 2550
type: B, layer: 1, pos: 2550
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 3267
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 3028
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2326
type: A, layer: 1, pos: 2326
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
type: B, layer: 1, pos: 413

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2431861, upper bound: 0.2435186
time: 19.03 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2435397, upper bound: 0.2435383
time: 60.27 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 85.65 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 85.65
Output dim: 3, lower bound: -0.2435143, upper bound: 0.2430129
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 85.65
Output dim: 3, lower bound: -0.2435385, upper bound: 0.2433678
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 85.65
Output dim: 3, lower bound: -0.2429892, upper bound: 0.2435138
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 85.65
Output dim: 3, lower bound: -0.2433401, upper bound: 0.2435379
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 85.65
Output dim: 3, lower bound: -0.2431861, upper bound: 0.2435186
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 85.65
Output dim: 3, lower bound: -0.2435397, upper bound: 0.2435383

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.0282471, -1.6372809, -3.0312083, -1.6357696, -0.9864225, 0.9817725
1: -2.3442659, -1.2733958, -2.3453226, -1.2738607, -0.5847844, 0.5815887
2: -1.0439389, -0.0099028, -1.0455842, -0.0088062, -0.8780749, 0.8788537
3: -0.4807271, 0.1627780, -0.4824075, 0.1634386, -0.5974931, 0.5984942
4: -1.9479333, -0.7919939, -1.9456735, -0.7914720, -0.6326704, 0.6326585
5: -0.6537327, 0.0325701, -0.6552645, 0.0333118, -0.6271425, 0.6279160
6: -1.0617039, -0.2353588, -1.0626569, -0.2345423, -0.5120683, 0.5134845
7: -2.2448490, -0.7772650, -2.2447343, -0.7769483, -0.8768479, 0.8774545
8: -1.3512719, -0.6119194, -1.3517051, -0.6118422, -0.3810863, 0.3780122
9: -1.1411247, -0.3275366, -1.1417470, -0.3273001, -0.4161119, 0.4166582

Time for backsubstitution: 6.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2630
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 3100
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 3036
type: B, layer: 1, pos: 2402
type: A, layer: 1, pos: 2402
type: B, layer: 1, pos: 3032
type: A, layer: 1, pos: 3032
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2584
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 2185
type: B, layer: 1, pos: 295
type: A, layer: 1, pos: 295
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2421
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2587
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3022
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 3037
type: B, layer: 1, pos: 3037
type: A, layer: 1, pos: 2642
type: B, layer: 1, pos: 2642
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 3441
type: A, layer: 1, pos: 3097
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 423
type: A, layer: 1, pos: 423
type: B, layer: 1, pos: 2422
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2854
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 2854
type: A, layer: 1, pos: 2327
type: B, layer: 1, pos: 2992
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2441
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2212
type: A, layer: 1, pos: 2212
type: B, layer: 1, pos: 2991
type: A, layer: 1, pos: 2511
type: B, layer: 1, pos: 2511
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2991
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 2417
type: A, layer: 1, pos: 3423
type: B, layer: 1, pos: 3423
type: B, layer: 1, pos: 413
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 3429
type: B, layer: 1, pos: 3429
type: B, layer: 1, pos: 3008
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2616
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 426
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2853
type: B, layer: 1, pos: 2853
type: B, layer: 1, pos: 2317
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2522
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 2526
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 3436
type: B, layer: 1, pos: 3436
type: B, layer: 1, pos: 2380
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 2600
type: A, layer: 1, pos: 3452
type: B, layer: 1, pos: 3452
type: A, layer: 1, pos: 2060
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 3259
type: B, layer: 1, pos: 3259
type: A, layer: 1, pos: 2873
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 2863
type: A, layer: 1, pos: 2863
type: B, layer: 1, pos: 279
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 279
type: B, layer: 1, pos: 2423
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 2079
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 3287
type: B, layer: 1, pos: 3275
type: A, layer: 1, pos: 3275
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 3320
type: B, layer: 1, pos: 3320
type: A, layer: 1, pos: 500
type: B, layer: 1, pos: 500
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2753
type: A, layer: 1, pos: 3321
type: B, layer: 1, pos: 3321
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 2753
type: A, layer: 1, pos: 3187
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 3319
type: B, layer: 1, pos: 3319
type: B, layer: 1, pos: 2400
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 3263
type: B, layer: 1, pos: 3263
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2160
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2097
type: A, layer: 1, pos: 2097
type: B, layer: 1, pos: 2394
type: A, layer: 1, pos: 3232
type: B, layer: 1, pos: 3232
type: A, layer: 1, pos: 3090
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2250
type: A, layer: 1, pos: 2250
type: B, layer: 1, pos: 2815
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 2815
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2577
type: A, layer: 1, pos: 2791
type: B, layer: 1, pos: 2791
type: A, layer: 1, pos: 3271
type: B, layer: 1, pos: 3271
type: B, layer: 1, pos: 3056
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2423
type: B, layer: 1, pos: 2866
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 2139
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 2095
type: A, layer: 1, pos: 2589
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3422
type: B, layer: 1, pos: 3422
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 2580
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3339
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3339
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 506
type: B, layer: 1, pos: 506
type: A, layer: 1, pos: 2394
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2352
type: A, layer: 1, pos: 3057
type: B, layer: 1, pos: 2771
type: A, layer: 1, pos: 2352
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 3565
type: B, layer: 1, pos: 3565
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 3457
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 3286
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 3028
type: B, layer: 1, pos: 2550
type: A, layer: 1, pos: 3564
type: A, layer: 1, pos: 2550
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3267
type: B, layer: 1, pos: 3267
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2926
type: B, layer: 1, pos: 3028
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2326
type: B, layer: 1, pos: 2326
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
type: A, layer: 1, pos: 2358

## Relational analysis of NS_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2433846, upper bound: 0.2428433
time: 77.16 seconds

## Relational analysis of NS_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2433853, upper bound: 0.2428873
time: 11.71 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.0323515, -1.6305423, -3.0328519, -1.6305342, -0.9962544, 0.9906202
1: -2.3454242, -1.2734928, -2.3462200, -1.2734921, -0.5866137, 0.5837066
2: -1.0462164, -0.0027980, -1.0462282, -0.0027985, -0.8861601, 0.8838528
3: -0.4829035, 0.1663917, -0.4829174, 0.1664192, -0.6027856, 0.6016617
4: -1.9463532, -0.7915238, -1.9463603, -0.7912142, -0.6372769, 0.6361361
5: -0.6559492, 0.0362687, -0.6559652, 0.0362930, -0.6325774, 0.6319669
6: -1.0627687, -0.2342762, -1.0627743, -0.2343005, -0.5133759, 0.5142508
7: -2.2466152, -0.7759120, -2.2466364, -0.7757989, -0.8819520, 0.8826101
8: -1.3521776, -0.6113950, -1.3524387, -0.6113905, -0.3817858, 0.3800803
9: -1.1416695, -0.3270905, -1.1419897, -0.3270957, -0.4168018, 0.4169059

Time for backsubstitution: 6.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2630
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 3100
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 3036
type: A, layer: 1, pos: 3036
type: A, layer: 1, pos: 2402
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 3032
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 2584
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 295
type: B, layer: 1, pos: 295
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 2421
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2587
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3022
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 3037
type: B, layer: 1, pos: 3037
type: B, layer: 1, pos: 2642
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 3441
type: A, layer: 1, pos: 3097
type: B, layer: 1, pos: 3097
type: A, layer: 1, pos: 423
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 2422
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2854
type: B, layer: 1, pos: 2854
type: B, layer: 1, pos: 2327
type: A, layer: 1, pos: 2327
type: B, layer: 1, pos: 2992
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 413
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2212
type: B, layer: 1, pos: 2212
type: B, layer: 1, pos: 2991
type: A, layer: 1, pos: 2991
type: A, layer: 1, pos: 2511
type: B, layer: 1, pos: 2511
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3423
type: A, layer: 1, pos: 3423
type: A, layer: 1, pos: 2616
type: B, layer: 1, pos: 2616
type: A, layer: 1, pos: 3429
type: B, layer: 1, pos: 3429
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 426
type: B, layer: 1, pos: 2417
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 426
type: A, layer: 1, pos: 2317
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 2853
type: A, layer: 1, pos: 2853
type: B, layer: 1, pos: 2600
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2522
type: A, layer: 1, pos: 2522
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 2526
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 3436
type: A, layer: 1, pos: 3436
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 3070
type: A, layer: 1, pos: 2380
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 3452
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 3452
type: B, layer: 1, pos: 2060
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 3259
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 427
type: A, layer: 1, pos: 2873
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 2863
type: A, layer: 1, pos: 2863
type: B, layer: 1, pos: 279
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 279
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 2079
type: B, layer: 1, pos: 2079
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 3287
type: B, layer: 1, pos: 3275
type: A, layer: 1, pos: 3275
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 3320
type: A, layer: 1, pos: 3320
type: A, layer: 1, pos: 500
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 3321
type: A, layer: 1, pos: 3321
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 2753
type: A, layer: 1, pos: 2753
type: B, layer: 1, pos: 3502
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 3187
type: B, layer: 1, pos: 3187
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 2423
type: B, layer: 1, pos: 3319
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 2400
type: B, layer: 1, pos: 2400
type: A, layer: 1, pos: 3263
type: B, layer: 1, pos: 3263
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 2547
type: B, layer: 1, pos: 2547
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 2097
type: A, layer: 1, pos: 2097
type: B, layer: 1, pos: 3232
type: A, layer: 1, pos: 3232
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 3090
type: B, layer: 1, pos: 3090
type: A, layer: 1, pos: 2250
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 2815
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2815
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2394
type: B, layer: 1, pos: 2791
type: A, layer: 1, pos: 2791
type: A, layer: 1, pos: 3271
type: B, layer: 1, pos: 3271
type: A, layer: 1, pos: 2866
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2866
type: A, layer: 1, pos: 2394
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 2139
type: B, layer: 1, pos: 2139
type: A, layer: 1, pos: 3453
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 2668
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 2668
type: A, layer: 1, pos: 2095
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3422
type: A, layer: 1, pos: 2577
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3422
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 2580
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 2589
type: A, layer: 1, pos: 3056
type: B, layer: 1, pos: 3339
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 506
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 2352
type: A, layer: 1, pos: 2352
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2771
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3565
type: A, layer: 1, pos: 3565
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 3457
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 3286
type: B, layer: 1, pos: 3286
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 2550
type: A, layer: 1, pos: 3267
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 3028
type: A, layer: 1, pos: 2326
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

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2358

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2434210, upper bound: 0.2432086
time: 360.83 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2434099, upper bound: 0.2432355
time: 156.99 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -3.0279434, -1.6289897, -3.0246489, -1.6392386, -0.9812666, 0.9910229
1: -2.3401976, -1.2697786, -2.3388479, -1.2748351, -0.5782124, 0.5852652
2: -1.0462189, -0.0089415, -1.0436263, -0.0100720, -0.8793947, 0.8772128
3: -0.4876202, 0.1632572, -0.4786640, 0.1625524, -0.6034553, 0.5951910
4: -1.9477820, -0.7916398, -1.9474858, -0.7919848, -0.6346743, 0.6319714
5: -0.6605533, 0.0331973, -0.6517453, 0.0324517, -0.6330000, 0.6248766
6: -1.0654622, -0.2347220, -1.0603423, -0.2356523, -0.5154982, 0.5109224
7: -2.2493291, -0.7773348, -2.2431867, -0.7776787, -0.8816498, 0.8753725
8: -1.3498647, -0.6094474, -1.3493855, -0.6131442, -0.3780280, 0.3824878
9: -1.1392682, -0.3270253, -1.1384051, -0.3275829, -0.4138310, 0.4138163

Time for backsubstitution: 6.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2630
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3100
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 3036
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2402
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 3032
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2584
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 2185
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 295
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2421
type: B, layer: 1, pos: 2587
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2587
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3037
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3037
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 3441
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 3097
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 423
type: B, layer: 1, pos: 423
type: A, layer: 1, pos: 2422
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2854
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 2854
type: B, layer: 1, pos: 2327
type: A, layer: 1, pos: 2992
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 2992
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2212
type: B, layer: 1, pos: 2212
type: A, layer: 1, pos: 2991
type: B, layer: 1, pos: 2616
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2417
type: B, layer: 1, pos: 2991
type: A, layer: 1, pos: 413
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 3423
type: A, layer: 1, pos: 3423
type: B, layer: 1, pos: 3429
type: A, layer: 1, pos: 3429
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 3008
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 426
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2853
type: A, layer: 1, pos: 2853
type: B, layer: 1, pos: 2417
type: A, layer: 1, pos: 2317
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2522
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2526
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 2526
type: A, layer: 1, pos: 2380
type: B, layer: 1, pos: 3436
type: A, layer: 1, pos: 3436
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 3452
type: A, layer: 1, pos: 3452
type: B, layer: 1, pos: 2060
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 3259
type: A, layer: 1, pos: 3259
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 2600
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 2423
type: B, layer: 1, pos: 3070
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 279
type: B, layer: 1, pos: 2863
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 2627
type: B, layer: 1, pos: 279
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 427
type: A, layer: 1, pos: 2079
type: B, layer: 1, pos: 2079
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 3275
type: B, layer: 1, pos: 3275
type: B, layer: 1, pos: 3320
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 3320
type: B, layer: 1, pos: 500
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 2753
type: B, layer: 1, pos: 3321
type: A, layer: 1, pos: 3321
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 2627
type: A, layer: 1, pos: 2753
type: B, layer: 1, pos: 3187
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 3319
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 2400
type: B, layer: 1, pos: 2400
type: B, layer: 1, pos: 3263
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 2328
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2547
type: A, layer: 1, pos: 2394
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2160
type: A, layer: 1, pos: 2097
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 3232
type: A, layer: 1, pos: 3232
type: B, layer: 1, pos: 3090
type: A, layer: 1, pos: 2815
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2250
type: B, layer: 1, pos: 2250
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3056
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2815
type: B, layer: 1, pos: 2791
type: A, layer: 1, pos: 2791
type: B, layer: 1, pos: 3271
type: A, layer: 1, pos: 3271
type: B, layer: 1, pos: 2866
type: A, layer: 1, pos: 2866
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 2589
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 2139
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3422
type: A, layer: 1, pos: 3422
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3339
type: B, layer: 1, pos: 3339
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 506
type: A, layer: 1, pos: 506
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 2352
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 2771
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 3565
type: A, layer: 1, pos: 3565
type: B, layer: 1, pos: 3457
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 3286
type: B, layer: 1, pos: 3286
type: A, layer: 1, pos: 2589
type: B, layer: 1, pos: 2423
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 3057
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 2394
type: B, layer: 1, pos: 3267
type: A, layer: 1, pos: 3267
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 3056
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 3028
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
type: B, layer: 1, pos: 2328

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2358

## Relational analysis of NS_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2428232, upper bound: 0.2433883
time: 254.55 seconds

## Relational analysis of NS_A2_B1_B1_B2

### Relational analysis result of NS_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2428659, upper bound: 0.2433851
time: 372.07 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -3.0295775, -1.6237555, -3.0287535, -1.6325011, -0.9901159, 1.0008507
1: -2.3410966, -1.2694098, -2.3400059, -1.2749320, -0.5803354, 0.5870957
2: -1.0468616, -0.0029339, -1.0459051, -0.0029671, -0.8843939, 0.8852988
3: -0.4881294, 0.1662375, -0.4808420, 0.1661657, -0.6066228, 0.6004866
4: -1.9484695, -0.7913818, -1.9459043, -0.7915150, -0.6381541, 0.6365785
5: -0.6612542, 0.0361785, -0.6539631, 0.0361499, -0.6370527, 0.6303133
6: -1.0655787, -0.2344798, -1.0614080, -0.2345697, -0.5162638, 0.5122304
7: -2.2512281, -0.7761853, -2.2449532, -0.7763259, -0.8868072, 0.8804749
8: -1.3505981, -0.6089963, -1.3502913, -0.6126193, -0.3800969, 0.3831872
9: -1.1395129, -0.3268209, -1.1389513, -0.3271370, -0.4140799, 0.4145063

Time for backsubstitution: 6.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2630
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3100
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 3036
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3036
type: A, layer: 1, pos: 2402
type: B, layer: 1, pos: 2402
type: A, layer: 1, pos: 3032
type: B, layer: 1, pos: 3032
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2584
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 2185
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 295
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2421
type: B, layer: 1, pos: 2587
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 3022
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 3037
type: A, layer: 1, pos: 3037
type: B, layer: 1, pos: 2642
type: A, layer: 1, pos: 2642
type: B, layer: 1, pos: 3441
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 3097
type: A, layer: 1, pos: 3097
type: B, layer: 1, pos: 423
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 2422
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2854
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 2854
type: B, layer: 1, pos: 2327
type: A, layer: 1, pos: 2992
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 413
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2212
type: B, layer: 1, pos: 2212
type: A, layer: 1, pos: 2991
type: B, layer: 1, pos: 2511
type: B, layer: 1, pos: 2991
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2616
type: A, layer: 1, pos: 3423
type: B, layer: 1, pos: 3423
type: B, layer: 1, pos: 3429
type: A, layer: 1, pos: 3429
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 3008
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2317
type: A, layer: 1, pos: 426
type: B, layer: 1, pos: 2853
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 2522
type: B, layer: 1, pos: 2522
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 2526
type: B, layer: 1, pos: 2526
type: A, layer: 1, pos: 3070
type: B, layer: 1, pos: 2600
type: A, layer: 1, pos: 3436
type: B, layer: 1, pos: 3436
type: A, layer: 1, pos: 2380
type: B, layer: 1, pos: 2380
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 3452
type: B, layer: 1, pos: 3452
type: B, layer: 1, pos: 2060
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 3070
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 3259
type: A, layer: 1, pos: 3259
type: B, layer: 1, pos: 2873
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 279
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 279
type: A, layer: 1, pos: 2079
type: B, layer: 1, pos: 2079
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2423
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 3275
type: B, layer: 1, pos: 3275
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 3320
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 500
type: A, layer: 1, pos: 500
type: B, layer: 1, pos: 2627
type: A, layer: 1, pos: 3321
type: B, layer: 1, pos: 3321
type: B, layer: 1, pos: 2753
type: A, layer: 1, pos: 2753
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 3187
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 3187
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 3319
type: B, layer: 1, pos: 3319
type: B, layer: 1, pos: 2400
type: A, layer: 1, pos: 2400
type: B, layer: 1, pos: 3263
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 2547
type: A, layer: 1, pos: 2547
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2160
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 2328
type: B, layer: 1, pos: 2423
type: A, layer: 1, pos: 2097
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 427
type: A, layer: 1, pos: 3232
type: B, layer: 1, pos: 3232
type: B, layer: 1, pos: 3090
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2394
type: B, layer: 1, pos: 2250
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 2815
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2815
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2791
type: B, layer: 1, pos: 2791
type: B, layer: 1, pos: 3271
type: A, layer: 1, pos: 3271
type: B, layer: 1, pos: 2866
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 2866
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 2139
type: A, layer: 1, pos: 2139
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 2095
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 2668
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 2668
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2589
type: A, layer: 1, pos: 3422
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 3422
type: B, layer: 1, pos: 2394
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3339
type: A, layer: 1, pos: 2580
type: B, layer: 1, pos: 2577
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 506
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 2589
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 3057
type: A, layer: 1, pos: 2771
type: B, layer: 1, pos: 2352
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 2771
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 3286
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 3286
type: B, layer: 1, pos: 2550
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 2550
type: B, layer: 1, pos: 3267
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 3028
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2326
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
type: B, layer: 1, pos: 2358

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2431900, upper bound: 0.2434201
time: 106.58 seconds

## Relational analysis of NS_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2432190, upper bound: 0.2434145
time: 18.17 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -3.0313418, -1.6289892, -3.0288823, -1.6372814, -0.9875532, 0.9895933
1: -2.3446350, -1.2697732, -2.3443112, -1.2733951, -0.5849661, 0.5869520
2: -1.0462762, -0.0088237, -1.0439322, -0.0099270, -0.8795651, 0.8781815
3: -0.4877300, 0.1634483, -0.4807414, 0.1627920, -0.6037832, 0.5975111
4: -1.9478421, -0.7915119, -1.9479380, -0.7918283, -0.6348453, 0.6327821
5: -0.6606799, 0.0332900, -0.6537538, 0.0325688, -0.6332760, 0.6271386
6: -1.0655037, -0.2344977, -1.0617114, -0.2353810, -0.5151302, 0.5129331
7: -2.2494836, -0.7770078, -2.2448721, -0.7772868, -0.8815001, 0.8775049
8: -1.3516873, -0.6094313, -1.3516064, -0.6119292, -0.3810395, 0.3816219
9: -1.1415553, -0.3269935, -1.1411488, -0.3275418, -0.4166313, 0.4170310

Time for backsubstitution: 6.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2630
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3100
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 3036
type: B, layer: 1, pos: 3036
type: A, layer: 1, pos: 2402
type: B, layer: 1, pos: 2402
type: A, layer: 1, pos: 3032
type: B, layer: 1, pos: 3032
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2584
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 2185
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 295
type: B, layer: 1, pos: 295
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2421
type: B, layer: 1, pos: 2587
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 3022
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 3037
type: A, layer: 1, pos: 3037
type: B, layer: 1, pos: 2642
type: A, layer: 1, pos: 2642
type: B, layer: 1, pos: 3441
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 3097
type: A, layer: 1, pos: 3097
type: A, layer: 1, pos: 423
type: B, layer: 1, pos: 423
type: A, layer: 1, pos: 2422
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2854
type: A, layer: 1, pos: 2854
type: A, layer: 1, pos: 2327
type: B, layer: 1, pos: 2327
type: A, layer: 1, pos: 2992
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2212
type: B, layer: 1, pos: 2212
type: A, layer: 1, pos: 2991
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2511
type: A, layer: 1, pos: 2511
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 3423
type: A, layer: 1, pos: 3423
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 3429
type: A, layer: 1, pos: 3429
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 413
type: B, layer: 1, pos: 2417
type: A, layer: 1, pos: 426
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 2853
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 2522
type: A, layer: 1, pos: 2522
type: B, layer: 1, pos: 2600
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 2526
type: B, layer: 1, pos: 2526
type: B, layer: 1, pos: 3436
type: A, layer: 1, pos: 3436
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 2380
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 3452
type: A, layer: 1, pos: 3452
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 2060
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 3259
type: A, layer: 1, pos: 3259
type: B, layer: 1, pos: 2873
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 2863
type: B, layer: 1, pos: 2863
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 279
type: B, layer: 1, pos: 279
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 2079
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 427
type: A, layer: 1, pos: 2627
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 3275
type: B, layer: 1, pos: 3275
type: A, layer: 1, pos: 3287
type: B, layer: 1, pos: 3320
type: A, layer: 1, pos: 3320
type: B, layer: 1, pos: 500
type: A, layer: 1, pos: 500
type: B, layer: 1, pos: 2627
type: A, layer: 1, pos: 2423
type: B, layer: 1, pos: 3321
type: A, layer: 1, pos: 3321
type: B, layer: 1, pos: 2753
type: A, layer: 1, pos: 2753
type: B, layer: 1, pos: 3502
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 3187
type: A, layer: 1, pos: 3187
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 3319
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 2400
type: B, layer: 1, pos: 2400
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 3263
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2547
type: A, layer: 1, pos: 2547
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2160
type: A, layer: 1, pos: 2097
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 3232
type: A, layer: 1, pos: 3232
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 3090
type: A, layer: 1, pos: 3090
type: A, layer: 1, pos: 2250
type: B, layer: 1, pos: 2250
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 2815
type: B, layer: 1, pos: 2815
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2394
type: B, layer: 1, pos: 2791
type: A, layer: 1, pos: 2791
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3271
type: B, layer: 1, pos: 3271
type: A, layer: 1, pos: 2866
type: B, layer: 1, pos: 2866
type: A, layer: 1, pos: 2577
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 2139
type: A, layer: 1, pos: 2139
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 2394
type: A, layer: 1, pos: 3453
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 2095
type: B, layer: 1, pos: 2565
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 2668
type: B, layer: 1, pos: 2668
type: A, layer: 1, pos: 3056
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3422
type: A, layer: 1, pos: 3422
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: B, layer: 1, pos: 3056
type: A, layer: 1, pos: 3339
type: B, layer: 1, pos: 3339
type: A, layer: 1, pos: 2589
type: B, layer: 1, pos: 506
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 2352
type: B, layer: 1, pos: 2352
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 2771
type: B, layer: 1, pos: 2771
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3565
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 3457
type: B, layer: 1, pos: 3457
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 3286
type: B, layer: 1, pos: 3286
type: A, layer: 1, pos: 2550
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 2550
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3028
type: A, layer: 1, pos: 3267
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2326
type: A, layer: 1, pos: 2326
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
type: B, layer: 1, pos: 2358

## Relational analysis of NS_A2_B2_B1_B1

### Relational analysis result of NS_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2430148, upper bound: 0.2433892
time: 206.37 seconds

## Relational analysis of NS_A2_B2_B1_B2

### Relational analysis result of NS_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2430560, upper bound: 0.2433892
time: 22.71 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -3.0329731, -1.6237535, -3.0329938, -1.6305428, -0.9964044, 0.9994213
1: -2.3455338, -1.2694043, -2.3454690, -1.2734921, -0.5870856, 0.5887815
2: -1.0469190, -0.0028160, -1.0462097, -0.0028223, -0.8845637, 0.8862668
3: -0.4882393, 0.1664292, -0.4829183, 0.1664060, -0.6069507, 0.6028036
4: -1.9485291, -0.7912540, -1.9463581, -0.7913580, -0.6383243, 0.6373891
5: -0.6613804, 0.0362716, -0.6559706, 0.0362678, -0.6373283, 0.6325740
6: -1.0656193, -0.2342558, -1.0627778, -0.2342985, -0.5158963, 0.5142408
7: -2.2513833, -0.7758585, -2.2466393, -0.7759337, -0.8866576, 0.8826098
8: -1.3524210, -0.6089805, -1.3525122, -0.6114047, -0.3831068, 0.3823216
9: -1.1417991, -0.3267892, -1.1416934, -0.3270960, -0.4168797, 0.4177210

Time for backsubstitution: 6.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2630
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 3100
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 3036
type: A, layer: 1, pos: 3036
type: B, layer: 1, pos: 2402
type: A, layer: 1, pos: 2402
type: B, layer: 1, pos: 3032
type: A, layer: 1, pos: 3032
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2584
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 2185
type: B, layer: 1, pos: 295
type: A, layer: 1, pos: 295
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2421
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2587
type: B, layer: 1, pos: 2587
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3022
type: B, layer: 1, pos: 3022
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3037
type: B, layer: 1, pos: 3037
type: A, layer: 1, pos: 2642
type: B, layer: 1, pos: 2642
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 3441
type: A, layer: 1, pos: 3097
type: B, layer: 1, pos: 3097
type: B, layer: 1, pos: 423
type: A, layer: 1, pos: 423
type: B, layer: 1, pos: 2422
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2854
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 2854
type: A, layer: 1, pos: 2327
type: B, layer: 1, pos: 2992
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 413
type: A, layer: 1, pos: 2441
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2212
type: A, layer: 1, pos: 2212
type: B, layer: 1, pos: 2991
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 2991
type: B, layer: 1, pos: 2511
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 3423
type: B, layer: 1, pos: 3423
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3429
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 3429
type: B, layer: 1, pos: 3008
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 2600
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2853
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 2522
type: B, layer: 1, pos: 2522
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 2526
type: A, layer: 1, pos: 2526
type: A, layer: 1, pos: 3436
type: B, layer: 1, pos: 3436
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 2380
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 3452
type: B, layer: 1, pos: 3452
type: A, layer: 1, pos: 2060
type: B, layer: 1, pos: 2060
type: A, layer: 1, pos: 3070
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 3259
type: B, layer: 1, pos: 3259
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2873
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 2863
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 279
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 279
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 2079
type: B, layer: 1, pos: 2627
type: A, layer: 1, pos: 2079
type: B, layer: 1, pos: 2423
type: A, layer: 1, pos: 3287
type: B, layer: 1, pos: 3275
type: A, layer: 1, pos: 3275
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 3320
type: B, layer: 1, pos: 3320
type: A, layer: 1, pos: 500
type: B, layer: 1, pos: 500
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 3321
type: A, layer: 1, pos: 2753
type: B, layer: 1, pos: 3321
type: B, layer: 1, pos: 2753
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 3187
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 3319
type: B, layer: 1, pos: 3319
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 2400
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 3263
type: B, layer: 1, pos: 3263
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2547
type: B, layer: 1, pos: 2547
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2160
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 2423
type: B, layer: 1, pos: 2097
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 3232
type: B, layer: 1, pos: 3232
type: A, layer: 1, pos: 3090
type: B, layer: 1, pos: 3090
type: B, layer: 1, pos: 2250
type: A, layer: 1, pos: 2250
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2394
type: A, layer: 1, pos: 2815
type: B, layer: 1, pos: 2815
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 2791
type: B, layer: 1, pos: 2791
type: B, layer: 1, pos: 3271
type: A, layer: 1, pos: 3271
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2866
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 2139
type: B, layer: 1, pos: 2139
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3453
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: B, layer: 1, pos: 2095
type: A, layer: 1, pos: 2577
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 2565
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 2668
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2394
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 3422
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: B, layer: 1, pos: 3422
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 506
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 2352
type: A, layer: 1, pos: 2352
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 2771
type: A, layer: 1, pos: 2771
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3057
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 3565
type: A, layer: 1, pos: 3565
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 3057
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 3457
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 3286
type: A, layer: 1, pos: 3286
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3564
type: A, layer: 1, pos: 2550
type: B, layer: 1, pos: 3267
type: A, layer: 1, pos: 3267
type: B, layer: 1, pos: 3028
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2326
type: B, layer: 1, pos: 2326
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
type: A, layer: 1, pos: 2358

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2434195, upper bound: 0.2433825
time: 136.15 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2434119, upper bound: 0.2434140
time: 18.17 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 160.97 seconds
NS_A1_B2_A1_A1, status: Status.VERIFIED, split count: 4, time: 160.97
Output dim: 3, lower bound: -0.2433846, upper bound: 0.2428433
NS_A1_B2_A1_A2, status: Status.VERIFIED, split count: 4, time: 160.97
Output dim: 3, lower bound: -0.2433853, upper bound: 0.2428873
NS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 160.97
Output dim: 3, lower bound: -0.2434210, upper bound: 0.2432086
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 160.97
Output dim: 3, lower bound: -0.2434099, upper bound: 0.2432355
NS_A2_B1_B1_B1, status: Status.VERIFIED, split count: 4, time: 160.97
Output dim: 3, lower bound: -0.2428232, upper bound: 0.2433883
NS_A2_B1_B1_B2, status: Status.VERIFIED, split count: 4, time: 160.97
Output dim: 3, lower bound: -0.2428659, upper bound: 0.2433851
NS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 160.97
Output dim: 3, lower bound: -0.2431900, upper bound: 0.2434201
NS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 160.97
Output dim: 3, lower bound: -0.2432190, upper bound: 0.2434145
NS_A2_B2_B1_B1, status: Status.VERIFIED, split count: 4, time: 160.97
Output dim: 3, lower bound: -0.2430148, upper bound: 0.2433892
NS_A2_B2_B1_B2, status: Status.VERIFIED, split count: 4, time: 160.97
Output dim: 3, lower bound: -0.2430560, upper bound: 0.2433892
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 160.97
Output dim: 3, lower bound: -0.2434195, upper bound: 0.2433825
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 160.97
Output dim: 3, lower bound: -0.2434119, upper bound: 0.2434140

## BFS NS instance: NS_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -3.0346618, -1.6351948, -3.0326653, -1.6345530, -0.9874243, 0.9828663
1: -2.3457935, -1.2754468, -2.3460469, -1.2751678, -0.5805746, 0.5792938
2: -1.0460252, -0.0029405, -1.0460472, -0.0029130, -0.8856131, 0.8834171
3: -0.4805422, 0.1663934, -0.4808638, 0.1663872, -0.6003882, 0.5996196
4: -1.9460590, -0.7918975, -1.9460955, -0.7915148, -0.6365510, 0.6354023
5: -0.6536536, 0.0363068, -0.6539375, 0.0362770, -0.6302665, 0.6299814
6: -1.0617650, -0.2341424, -1.0619344, -0.2343092, -0.5122685, 0.5132649
7: -2.2444119, -0.7758743, -2.2446873, -0.7758021, -0.8795563, 0.8805486
8: -1.3542886, -0.6128235, -1.3524346, -0.6125493, -0.3770835, 0.3762957
9: -1.1407754, -0.3270712, -1.1418319, -0.3270975, -0.4158620, 0.4168754

Time for backsubstitution: 6.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2630
type: B, layer: 1, pos: 2630
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 3100
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3036
type: A, layer: 1, pos: 3036
type: A, layer: 1, pos: 2402
type: B, layer: 1, pos: 2402
type: B, layer: 1, pos: 3032
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 2584
type: B, layer: 1, pos: 2584
type: A, layer: 1, pos: 2185
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 295
type: B, layer: 1, pos: 295
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2421
type: B, layer: 1, pos: 2587
type: A, layer: 1, pos: 2587
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3022
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3018
type: A, layer: 1, pos: 3037
type: B, layer: 1, pos: 3037
type: B, layer: 1, pos: 2642
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 3441
type: A, layer: 1, pos: 3097
type: B, layer: 1, pos: 3097
type: A, layer: 1, pos: 423
type: B, layer: 1, pos: 423
type: A, layer: 1, pos: 2422
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2854
type: A, layer: 1, pos: 2854
type: B, layer: 1, pos: 2327
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 2992
type: B, layer: 1, pos: 2992
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 413
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2212
type: B, layer: 1, pos: 2212
type: A, layer: 1, pos: 2991
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2511
type: A, layer: 1, pos: 2511
type: B, layer: 1, pos: 3423
type: A, layer: 1, pos: 3423
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 2616
type: B, layer: 1, pos: 2616
type: A, layer: 1, pos: 3429
type: B, layer: 1, pos: 3429
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 3008
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 426
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 426
type: A, layer: 1, pos: 2317
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 2853
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 2522
type: A, layer: 1, pos: 2522
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 2526
type: A, layer: 1, pos: 2526
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 3436
type: A, layer: 1, pos: 3436
type: B, layer: 1, pos: 2380
type: A, layer: 1, pos: 2380
type: B, layer: 1, pos: 3070
type: A, layer: 1, pos: 3070
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 3452
type: A, layer: 1, pos: 3452
type: B, layer: 1, pos: 2060
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 3259
type: A, layer: 1, pos: 3259
type: B, layer: 1, pos: 427
type: A, layer: 1, pos: 2873
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 2863
type: A, layer: 1, pos: 2863
type: B, layer: 1, pos: 279
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 279
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 2079
type: B, layer: 1, pos: 2079
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 3287
type: B, layer: 1, pos: 3275
type: A, layer: 1, pos: 3275
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3320
type: A, layer: 1, pos: 3320
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 500
type: A, layer: 1, pos: 500
type: B, layer: 1, pos: 3321
type: A, layer: 1, pos: 3321
type: B, layer: 1, pos: 2753
type: A, layer: 1, pos: 2753
type: B, layer: 1, pos: 3502
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 3187
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 2423
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 3319
type: A, layer: 1, pos: 3319
type: A, layer: 1, pos: 2400
type: B, layer: 1, pos: 2400
type: A, layer: 1, pos: 3263
type: B, layer: 1, pos: 3263
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 2547
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 3038
type: A, layer: 1, pos: 2097
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 3232
type: A, layer: 1, pos: 3232
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 3090
type: B, layer: 1, pos: 3090
type: A, layer: 1, pos: 2250
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 2815
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 2815
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2328
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2791
type: A, layer: 1, pos: 2791
type: A, layer: 1, pos: 3271
type: B, layer: 1, pos: 3271
type: B, layer: 1, pos: 2394
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2394
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 2139
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 3453
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 2668
type: A, layer: 1, pos: 2095
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 2668
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3056
type: A, layer: 1, pos: 2577
type: B, layer: 1, pos: 3422
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: A, layer: 1, pos: 3422
type: B, layer: 1, pos: 2589
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3056
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 2580
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2589
type: B, layer: 1, pos: 3339
type: A, layer: 1, pos: 506
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 2352
type: A, layer: 1, pos: 2352
type: B, layer: 1, pos: 2771
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 3057
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3565
type: A, layer: 1, pos: 3565
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 3457
type: B, layer: 1, pos: 3057
type: B, layer: 1, pos: 3457
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 3286
type: B, layer: 1, pos: 3286
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 2550
type: A, layer: 1, pos: 3267
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2326
type: B, layer: 1, pos: 3028
type: A, layer: 1, pos: 2326
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

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2630

## Relational analysis of NS_A1_B2_A2_A1_A1

### Relational analysis result of NS_A1_B2_A2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2433743, upper bound: 0.2430302
time: 40.25 seconds

## Relational analysis of NS_A1_B2_A2_A1_A2

### Relational analysis result of NS_A1_B2_A2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2433754, upper bound: 0.2431639
time: 39.63 seconds

## BFS NS instance: NS_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -3.0323062, -1.6314769, -3.0328159, -1.6313105, -0.9958621, 0.9811683
1: -2.3453975, -1.2744001, -2.3461990, -1.2742249, -0.5865389, 0.5773855
2: -1.0461254, -0.0028255, -1.0461553, -0.0028202, -0.8858932, 0.8833797
3: -0.4817607, 0.1663843, -0.4819905, 0.1664134, -0.6016040, 0.6006986
4: -1.9462919, -0.7915951, -1.9463112, -0.7912710, -0.6370237, 0.6356932
5: -0.6550699, 0.0362656, -0.6552110, 0.0362906, -0.6315285, 0.6310626
6: -1.0621567, -0.2342792, -1.0622822, -0.2343029, -0.5125387, 0.5138661
7: -2.2458746, -0.7759126, -2.2460306, -0.7757995, -0.8812315, 0.8820125
8: -1.3521771, -0.6126268, -1.3524382, -0.6124181, -0.3817138, 0.3747030
9: -1.1416489, -0.3270914, -1.1419735, -0.3270962, -0.4168029, 0.4168797

Time for backsubstitution: 6.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2630
type: A, layer: 1, pos: 2630
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 3100
type: B, layer: 1, pos: 3036
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 2402
type: A, layer: 1, pos: 2402
type: A, layer: 1, pos: 3036
type: A, layer: 1, pos: 3032
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2584
type: B, layer: 1, pos: 2584
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 295
type: B, layer: 1, pos: 295
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2421
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2587
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2587
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 3018
type: A, layer: 1, pos: 3037
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3037
type: A, layer: 1, pos: 2642
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 2642
type: B, layer: 1, pos: 3441
type: A, layer: 1, pos: 3097
type: B, layer: 1, pos: 3097
type: A, layer: 1, pos: 423
type: B, layer: 1, pos: 423
type: B, layer: 1, pos: 2422
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2854
type: B, layer: 1, pos: 2327
type: B, layer: 1, pos: 2854
type: A, layer: 1, pos: 2327
type: B, layer: 1, pos: 2992
type: A, layer: 1, pos: 2992
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 413
type: A, layer: 1, pos: 2441
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2212
type: A, layer: 1, pos: 2212
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 2511
type: B, layer: 1, pos: 2511
type: A, layer: 1, pos: 2991
type: B, layer: 1, pos: 3423
type: A, layer: 1, pos: 3423
type: B, layer: 1, pos: 2417
type: A, layer: 1, pos: 3429
type: B, layer: 1, pos: 3429
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 3008
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 2616
type: A, layer: 1, pos: 426
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 2417
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 426
type: A, layer: 1, pos: 2853
type: B, layer: 1, pos: 2853
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2317
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 2522
type: A, layer: 1, pos: 2522
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 2526
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2526
type: B, layer: 1, pos: 2380
type: B, layer: 1, pos: 3436
type: A, layer: 1, pos: 3436
type: A, layer: 1, pos: 2380
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 3452
type: B, layer: 1, pos: 3452
type: A, layer: 1, pos: 2060
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 3259
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 2600
type: A, layer: 1, pos: 2873
type: B, layer: 1, pos: 2873
type: A, layer: 1, pos: 3070
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 2863
type: B, layer: 1, pos: 279
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 2423
type: A, layer: 1, pos: 2863
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 528
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 279
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 2079
type: A, layer: 1, pos: 2079
type: A, layer: 1, pos: 3287
type: B, layer: 1, pos: 3275
type: A, layer: 1, pos: 3275
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 3320
type: A, layer: 1, pos: 3320
type: A, layer: 1, pos: 500
type: B, layer: 1, pos: 500
type: A, layer: 1, pos: 2627
type: B, layer: 1, pos: 3321
type: A, layer: 1, pos: 3321
type: A, layer: 1, pos: 2753
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 2753
type: A, layer: 1, pos: 3187
type: B, layer: 1, pos: 3187
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 3502
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 3319
type: A, layer: 1, pos: 3319
type: B, layer: 1, pos: 2328
type: B, layer: 1, pos: 2400
type: A, layer: 1, pos: 2400
type: A, layer: 1, pos: 3263
type: B, layer: 1, pos: 3263
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 805
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 2394
type: B, layer: 1, pos: 2160
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2097
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 427
type: B, layer: 1, pos: 3232
type: A, layer: 1, pos: 3232
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2815
type: A, layer: 1, pos: 3090
type: B, layer: 1, pos: 3090
type: A, layer: 1, pos: 2250
type: B, layer: 1, pos: 2250
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3056
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2815
type: B, layer: 1, pos: 2791
type: A, layer: 1, pos: 2791
type: A, layer: 1, pos: 3271
type: B, layer: 1, pos: 3271
type: A, layer: 1, pos: 2866
type: B, layer: 1, pos: 2866
type: A, layer: 1, pos: 2139
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 3453
type: B, layer: 1, pos: 2139
type: A, layer: 1, pos: 597
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 2589
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 3038
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 2668
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 2668
type: A, layer: 1, pos: 2095
type: B, layer: 1, pos: 3422
type: A, layer: 1, pos: 3422
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3339
type: B, layer: 1, pos: 2580
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 506
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 2140
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 3057
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 2352
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 2771
type: A, layer: 1, pos: 2352
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 2771
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2589
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 3028
type: A, layer: 1, pos: 2577
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3565
type: A, layer: 1, pos: 2394
type: A, layer: 1, pos: 3457
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3457
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 3286
type: B, layer: 1, pos: 3286
type: A, layer: 1, pos: 2550
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 2550
type: B, layer: 1, pos: 3057
type: A, layer: 1, pos: 3267
type: B, layer: 1, pos: 3267
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2326
type: A, layer: 1, pos: 3056
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
type: B, layer: 1, pos: 3028
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 2328

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2630

## Relational analysis of NS_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2432306, upper bound: 0.2431930
time: 299.75 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2433652, upper bound: 0.2431954
time: 12.26 seconds

## BFS NS instance: NS_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -3.0293930, -1.6277740, -3.0307205, -1.6372776, -0.9823616, 0.9916244
1: -2.3409238, -1.2710854, -2.3399482, -1.2768536, -0.5759193, 0.5804547
2: -1.0466807, -0.0030484, -1.0457118, -0.0031096, -0.8839580, 0.8847501
3: -0.4860790, 0.1662056, -0.4783393, 0.1661608, -0.6045763, 0.5979455
4: -1.9482052, -0.7916828, -1.9455938, -0.7918891, -0.6374192, 0.6358332
5: -0.6592277, 0.0361625, -0.6514480, 0.0361787, -0.6350585, 0.6278639
6: -1.0647370, -0.2344884, -1.0604185, -0.2344361, -0.5152299, 0.5111012
7: -2.2492809, -0.7761881, -2.2424779, -0.7762918, -0.8847433, 0.8778194
8: -1.3505943, -0.6101556, -1.3522248, -0.6141651, -0.3763096, 0.3784150
9: -1.1393569, -0.3268227, -1.1377912, -0.3271176, -0.4140440, 0.4132118

Time for backsubstitution: 6.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2630
type: B, layer: 1, pos: 2630
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 3100
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 3036
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3036
type: A, layer: 1, pos: 2402
type: B, layer: 1, pos: 2402
type: A, layer: 1, pos: 3032
type: B, layer: 1, pos: 3032
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2584
type: A, layer: 1, pos: 2584
type: A, layer: 1, pos: 2185
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 295
type: A, layer: 1, pos: 295
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2421
type: B, layer: 1, pos: 2587
type: A, layer: 1, pos: 2587
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 3022
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 3018
type: B, layer: 1, pos: 3018
type: B, layer: 1, pos: 3037
type: A, layer: 1, pos: 3037
type: B, layer: 1, pos: 2642
type: A, layer: 1, pos: 2642
type: B, layer: 1, pos: 3441
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 3097
type: A, layer: 1, pos: 3097
type: B, layer: 1, pos: 423
type: A, layer: 1, pos: 423
type: A, layer: 1, pos: 2422
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2854
type: A, layer: 1, pos: 2327
type: A, layer: 1, pos: 2854
type: B, layer: 1, pos: 2327
type: A, layer: 1, pos: 2992
type: B, layer: 1, pos: 2992
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 413
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2212
type: B, layer: 1, pos: 2212
type: A, layer: 1, pos: 2991
type: B, layer: 1, pos: 2991
type: B, layer: 1, pos: 2511
type: A, layer: 1, pos: 2511
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 2616
type: A, layer: 1, pos: 3423
type: B, layer: 1, pos: 3423
type: B, layer: 1, pos: 3429
type: A, layer: 1, pos: 3429
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 3008
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 2417
type: B, layer: 1, pos: 426
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2417
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 2317
type: A, layer: 1, pos: 426
type: B, layer: 1, pos: 2853
type: A, layer: 1, pos: 2853
type: A, layer: 1, pos: 2317
type: A, layer: 1, pos: 200
type: B, layer: 1, pos: 200
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2522
type: B, layer: 1, pos: 2522
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 2526
type: B, layer: 1, pos: 2600
type: B, layer: 1, pos: 2526
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 3436
type: B, layer: 1, pos: 3436
type: A, layer: 1, pos: 2380
type: B, layer: 1, pos: 2380
type: A, layer: 1, pos: 513
type: B, layer: 1, pos: 513
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 3452
type: B, layer: 1, pos: 3452
type: B, layer: 1, pos: 2060
type: A, layer: 1, pos: 2060
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 3259
type: A, layer: 1, pos: 3259
type: B, layer: 1, pos: 2873
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 427
type: A, layer: 1, pos: 2863
type: B, layer: 1, pos: 2863
type: A, layer: 1, pos: 279
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 279
type: B, layer: 1, pos: 528
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 2079
type: B, layer: 1, pos: 2079
type: A, layer: 1, pos: 2627
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 3275
type: B, layer: 1, pos: 3275
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 3320
type: B, layer: 1, pos: 3320
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 500
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 2423
type: A, layer: 1, pos: 3321
type: B, layer: 1, pos: 3321
type: B, layer: 1, pos: 2753
type: A, layer: 1, pos: 2753
type: B, layer: 1, pos: 3502
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 3187
type: A, layer: 1, pos: 3187
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 3319
type: B, layer: 1, pos: 3319
type: B, layer: 1, pos: 2400
type: A, layer: 1, pos: 2400
type: B, layer: 1, pos: 3263
type: A, layer: 1, pos: 3263
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: A, layer: 1, pos: 3038
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 2423
type: B, layer: 1, pos: 2547
type: A, layer: 1, pos: 2547
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 2160
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 2097
type: B, layer: 1, pos: 2097
type: A, layer: 1, pos: 2328
type: A, layer: 1, pos: 3232
type: B, layer: 1, pos: 427
type: B, layer: 1, pos: 3232
type: B, layer: 1, pos: 3038
type: B, layer: 1, pos: 3090
type: A, layer: 1, pos: 3090
type: B, layer: 1, pos: 2250
type: A, layer: 1, pos: 2250
type: A, layer: 1, pos: 2815
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 2394
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2815
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2791
type: B, layer: 1, pos: 2791
type: B, layer: 1, pos: 3271
type: A, layer: 1, pos: 3271
type: B, layer: 1, pos: 2866
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 3056
type: B, layer: 1, pos: 598
type: A, layer: 1, pos: 598
type: B, layer: 1, pos: 2139
type: A, layer: 1, pos: 2139
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 597
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 597
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 2668
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 2668
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2394
type: A, layer: 1, pos: 3422
type: B, layer: 1, pos: 150
type: A, layer: 1, pos: 150
type: B, layer: 1, pos: 3422
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3339
type: B, layer: 1, pos: 2580
type: A, layer: 1, pos: 2580
type: A, layer: 1, pos: 3339
type: A, layer: 1, pos: 2589
type: B, layer: 1, pos: 506
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 3056
type: A, layer: 1, pos: 819
type: B, layer: 1, pos: 819
type: A, layer: 1, pos: 2352
type: A, layer: 1, pos: 2140
type: B, layer: 1, pos: 2352
type: A, layer: 1, pos: 2771
type: B, layer: 1, pos: 3057
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 2771
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 522
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 3565
type: A, layer: 1, pos: 522
type: B, layer: 1, pos: 3565
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 3457
type: A, layer: 1, pos: 3457
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 526
type: B, layer: 1, pos: 526
type: A, layer: 1, pos: 3057
type: B, layer: 1, pos: 3028
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 2328
type: A, layer: 1, pos: 3286
type: B, layer: 1, pos: 2550
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 2550
type: B, layer: 1, pos: 3267
type: A, layer: 1, pos: 3267
type: A, layer: 1, pos: 751
type: B, layer: 1, pos: 751
type: A, layer: 1, pos: 2926
type: B, layer: 1, pos: 2926
type: A, layer: 1, pos: 2326
type: A, layer: 1, pos: 3028
type: B, layer: 1, pos: 2326
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
type: A, layer: 1, pos: 2630

## Relational analysis of NS_A2_B1_B2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2431428, upper bound: 0.2432426
time: 20.03 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2431438, upper bound: 0.2433766
time: 476.97 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 98.52 + 3689.81 = 3788.32 seconds
