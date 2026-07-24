## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 13)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.006293288199999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210558, 0.3210558)
1: (-1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5872045, 0.5872043)
2: (-3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1248990, 0.1248990)
3: (-4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5084342, 0.5084342)
4: (-2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2478142, 0.2478143)
5: (-5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4664164, 0.4664164)
6: (-5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3263453, 0.3263453)
7: (-2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4478140, 0.4478140)
8: (0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199811, 0.0199811)
9: (-0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0621305, 0.0621305)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.82 + 26.93 = 34.75 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0062995, upper bound: 0.0062978

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3475
type: B, layer: 1, pos: 3475
type: A, layer: 1, pos: 2198
type: B, layer: 1, pos: 2198
type: A, layer: 1, pos: 335
type: B, layer: 1, pos: 335
type: A, layer: 1, pos: 398
type: B, layer: 1, pos: 398
type: A, layer: 1, pos: 2197
type: B, layer: 1, pos: 2197
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 3081
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 388
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 373
type: A, layer: 1, pos: 2107
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 403
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 3082
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 400
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: A, layer: 1, pos: 2366
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: A, layer: 1, pos: 2799
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 2824
type: B, layer: 1, pos: 2824
type: A, layer: 1, pos: 2373
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 3076
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 2117
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: B, layer: 1, pos: 2830
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: A, layer: 1, pos: 352
type: B, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: B, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 2575
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: B, layer: 1, pos: 2095
type: A, layer: 1, pos: 3042
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842
type: B, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3475

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062289, upper bound: 0.0063006
time: 16.55 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062981, upper bound: 0.0063022
time: 58.14 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 74.78 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 74.78
Output dim: 8, lower bound: -0.0062289, upper bound: 0.0063006
NS_A2, status: Status.UNKNOWN, split count: 1, time: 74.78
Output dim: 8, lower bound: -0.0062981, upper bound: 0.0063022

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.1878206, 0.4343503, -0.1877884, 0.4343806, -0.3207370, 0.3207229
1: -1.4450300, -0.2965528, -1.4451116, -0.2964464, -0.5860035, 0.5858675
2: -3.2380939, -2.2599263, -3.2387242, -2.2594924, -0.1216176, 0.1217548
3: -4.2024016, -2.7086749, -4.2010441, -2.7085853, -0.5070263, 0.5064877
4: -2.8619604, -1.4610894, -2.8620722, -1.4609909, -0.2470093, 0.2470102
5: -5.2487602, -3.6403937, -5.2471595, -3.6402857, -0.4650736, 0.4641578
6: -5.8283653, -4.1405816, -5.8273864, -4.1405034, -0.3247032, 0.3247740
7: -2.8060203, -1.2579911, -2.8060412, -1.2577825, -0.4442911, 0.4440805
8: 0.9796933, 1.1550351, 0.9794137, 1.1554775, -0.0180265, 0.0178572
9: -0.0997037, 0.3797040, -0.0997216, 0.3797332, -0.0614240, 0.0614826

Time for backsubstitution: 5.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2198
type: B, layer: 1, pos: 2198
type: A, layer: 1, pos: 335
type: B, layer: 1, pos: 335
type: A, layer: 1, pos: 398
type: B, layer: 1, pos: 398
type: A, layer: 1, pos: 2197
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3081
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 373
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 403
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 3082
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 400
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: A, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2373
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 3076
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: B, layer: 1, pos: 2830
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: A, layer: 1, pos: 352
type: B, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 3032
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: A, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2842

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2198

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062264, upper bound: 0.0062405
time: 3.63 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062265, upper bound: 0.0062967
time: 10.15 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.1878182, 0.4345977, -0.1878242, 0.4345983, -0.3209770, 0.3207929
1: -1.4451118, -0.2953431, -1.4451118, -0.2953430, -0.5872023, 0.5858434
2: -3.2387247, -2.2567906, -3.2387249, -2.2567894, -0.1248947, 0.1212655
3: -4.2010441, -2.7083364, -4.2010441, -2.7082868, -0.5084275, 0.5065953
4: -2.8620729, -1.4602902, -2.8620734, -1.4602902, -0.2477962, 0.2469348
5: -5.2471590, -3.6399975, -5.2471595, -3.6399388, -0.4664082, 0.4645348
6: -5.8273864, -4.1402969, -5.8273873, -4.1402559, -0.3263385, 0.3244812
7: -2.8060417, -1.2545300, -2.8060415, -1.2545289, -0.4477519, 0.4439150
8: 0.9777473, 1.1554775, 0.9777470, 1.1554775, -0.0176366, 0.0199807
9: -0.0997217, 0.3802174, -0.0997217, 0.3802177, -0.0621294, 0.0613536

Time for backsubstitution: 6.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2198
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 335
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 398
type: B, layer: 1, pos: 398
type: A, layer: 1, pos: 2197
type: B, layer: 1, pos: 2197
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 3081
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 3080
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 388
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2166
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 373
type: A, layer: 1, pos: 373
type: A, layer: 1, pos: 2107
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 403
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 2374
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 3082
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 2809
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 401
type: B, layer: 1, pos: 3033
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 400
type: B, layer: 1, pos: 400
type: B, layer: 1, pos: 3276
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: B, layer: 1, pos: 3048
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 2799
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 2824
type: B, layer: 1, pos: 2824
type: A, layer: 1, pos: 2373
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 3489
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 3076
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 2117
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 2362
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 3274
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 352
type: B, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: B, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2842

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2198

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062962, upper bound: 0.0062396
time: 48.93 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062982, upper bound: 0.0062992
time: 90.56 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 145.57 seconds
NS_A1_A1, status: Status.VERIFIED, split count: 2, time: 145.57
Output dim: 8, lower bound: -0.0062264, upper bound: 0.0062405
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 145.57
Output dim: 8, lower bound: -0.0062265, upper bound: 0.0062967
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 145.57
Output dim: 8, lower bound: -0.0062962, upper bound: 0.0062396
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 145.57
Output dim: 8, lower bound: -0.0062982, upper bound: 0.0062992

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -0.1877563, 0.4343504, -0.1877321, 0.4343805, -0.3196204, 0.3206771
1: -1.4450265, -0.2965748, -1.4451087, -0.2964654, -0.5860004, 0.5849405
2: -3.2379844, -2.2599266, -3.2386284, -2.2594929, -0.1155499, 0.1217418
3: -4.2022600, -2.7086749, -4.2009201, -2.7085855, -0.5015232, 0.5064048
4: -2.8614314, -1.4610893, -2.8616085, -1.4609911, -0.2364327, 0.2469757
5: -5.2486563, -3.6403937, -5.2470675, -3.6402864, -0.4587620, 0.4641208
6: -5.8283281, -4.1405821, -5.8273535, -4.1405039, -0.3233338, 0.3247704
7: -2.8058729, -1.2579914, -2.8059118, -1.2577827, -0.4377021, 0.4440106
8: 0.9796933, 1.1550102, 0.9794137, 1.1554557, -0.0180229, 0.0166200
9: -0.0997033, 0.3796940, -0.0997213, 0.3797244, -0.0614203, 0.0612927

Time for backsubstitution: 6.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 335
type: A, layer: 1, pos: 335
type: B, layer: 1, pos: 398
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 2197
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3081
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 373
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 403
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: A, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3042
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 335

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062069, upper bound: 0.0063014
time: 3.52 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062273, upper bound: 0.0062967
time: 3.20 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -0.1869830, 0.4344653, -0.1870875, 0.4345976, -0.3201481, 0.3197346
1: -1.4440718, -0.2961222, -1.4450558, -0.2960308, -0.5857093, 0.5850408
2: -3.2345552, -2.2579494, -3.2350416, -2.2567899, -0.1202883, 0.1155765
3: -4.1965199, -2.7095208, -4.1970520, -2.7082906, -0.5038099, 0.5011306
4: -2.8553326, -1.4613158, -2.8561258, -1.4602909, -0.2397936, 0.2370479
5: -5.2420769, -3.6414380, -5.2426753, -3.6399403, -0.4611160, 0.4581527
6: -5.8263965, -4.1405487, -5.8265123, -4.1402569, -0.3252630, 0.3232008
7: -2.8007231, -1.2560701, -2.8013647, -1.2545307, -0.4423559, 0.4374027
8: 0.9780055, 1.1545877, 0.9777470, 1.1546928, -0.0164694, 0.0190498
9: -0.0996909, 0.3800770, -0.0997168, 0.3800929, -0.0619385, 0.0611902

Time for backsubstitution: 6.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 335
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 398
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 2197
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 3081
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 3080
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 388
type: B, layer: 1, pos: 2576
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2166
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 3475
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 373
type: A, layer: 1, pos: 373
type: A, layer: 1, pos: 2107
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 403
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 2374
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 3082
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 2809
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 401
type: B, layer: 1, pos: 3033
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 400
type: B, layer: 1, pos: 400
type: B, layer: 1, pos: 3276
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2366
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 3048
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 2799
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 2824
type: B, layer: 1, pos: 2824
type: A, layer: 1, pos: 2373
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 3489
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 3076
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 2117
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 2830
type: A, layer: 1, pos: 2830
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2386
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 2362
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 3274
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 352
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 2575
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2842

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 335

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062774, upper bound: 0.0062410
time: 5.57 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062960, upper bound: 0.0062411
time: 3.35 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -0.1877540, 0.4345976, -0.1877680, 0.4345982, -0.3198603, 0.3207469
1: -1.4451087, -0.2953646, -1.4451091, -0.2953620, -0.5871996, 0.5849167
2: -3.2386155, -2.2567911, -3.2386289, -2.2567897, -0.1188270, 0.1212524
3: -4.2009034, -2.7083364, -4.2009211, -2.7082865, -0.5029244, 0.5065120
4: -2.8615437, -1.4602904, -2.8616099, -1.4602902, -0.2372196, 0.2469003
5: -5.2470546, -3.6399975, -5.2470684, -3.6399388, -0.4600963, 0.4644979
6: -5.8273487, -4.1402969, -5.8273535, -4.1402559, -0.3249691, 0.3244777
7: -2.8058937, -1.2545302, -2.8059120, -1.2545290, -0.4411630, 0.4438449
8: 0.9777473, 1.1554527, 0.9777470, 1.1554558, -0.0176330, 0.0187435
9: -0.0997212, 0.3802075, -0.0997214, 0.3802091, -0.0621257, 0.0611637

Time for backsubstitution: 6.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 335
type: A, layer: 1, pos: 335
type: B, layer: 1, pos: 398
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 2197
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3081
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 373
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 403
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 400
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: A, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: B, layer: 1, pos: 2830
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3042
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 335

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062765, upper bound: 0.0062963
time: 4.74 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062973, upper bound: 0.0062996
time: 22.89 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 34.04 seconds
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 34.04
Output dim: 8, lower bound: -0.0062069, upper bound: 0.0063014
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 34.04
Output dim: 8, lower bound: -0.0062273, upper bound: 0.0062967
NS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 34.04
Output dim: 8, lower bound: -0.0062774, upper bound: 0.0062410
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 34.04
Output dim: 8, lower bound: -0.0062960, upper bound: 0.0062411
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 34.04
Output dim: 8, lower bound: -0.0062765, upper bound: 0.0062963
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 34.04
Output dim: 8, lower bound: -0.0062973, upper bound: 0.0062996

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1875358, 0.4341820, -0.1874891, 0.4341963, -0.3190774, 0.3201019
1: -1.4449971, -0.2965754, -1.4450756, -0.2964661, -0.5859474, 0.5848838
2: -3.2359483, -2.2599263, -3.2363648, -2.2594926, -0.1135668, 0.1195774
3: -4.2007680, -2.7086945, -4.1992860, -2.7086067, -0.4996389, 0.5044645
4: -2.8588989, -1.4610897, -2.8587925, -1.4609911, -0.2339824, 0.2443111
5: -5.2472858, -3.6403956, -5.2455683, -3.6402884, -0.4574474, 0.4627558
6: -5.8248014, -4.1405821, -5.8234310, -4.1405053, -0.3200326, 0.3212098
7: -2.8051107, -1.2579913, -2.8050809, -1.2577828, -0.4361169, 0.4422991
8: 0.9796933, 1.1547163, 0.9794137, 1.1551349, -0.0176657, 0.0162942
9: -0.0997034, 0.3796397, -0.0997212, 0.3796639, -0.0612416, 0.0611310

Time for backsubstitution: 6.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 398
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 2197
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3081
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 373
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 403
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: A, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 352
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3042
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 398

## Relational analysis of NS_A1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061385, upper bound: 0.0062994
time: 26.01 seconds

## Relational analysis of NS_A1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062059, upper bound: 0.0062975
time: 10.62 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1876956, 0.4343469, -0.1877787, 0.4343776, -0.3193735, 0.3205093
1: -1.4449584, -0.2965748, -1.4451256, -0.2963121, -0.5861091, 0.5850611
2: -3.2379637, -2.2599263, -3.2386422, -2.2560625, -0.1187984, 0.1197679
3: -4.2018929, -2.7086782, -4.2010531, -2.7059689, -0.5032661, 0.5057181
4: -2.8614058, -1.4610896, -2.8616085, -1.4567587, -0.2403902, 0.2445229
5: -5.2483072, -3.6403944, -5.2473044, -3.6380429, -0.4605045, 0.4637673
6: -5.8282566, -4.1405826, -5.8272810, -4.1346655, -0.3285695, 0.3214571
7: -2.8047605, -1.2579913, -2.8049283, -1.2571322, -0.4401961, 0.4427217
8: 0.9796933, 1.1549926, 0.9789096, 1.1554434, -0.0176905, 0.0171643
9: -0.0997035, 0.3795182, -0.0996822, 0.3795412, -0.0612464, 0.0615639

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 398
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 2197
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 388
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: A, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 3076
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2386
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 352
type: A, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2095
type: A, layer: 1, pos: 3042
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 2095
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2361

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 398

## Relational analysis of NS_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061583, upper bound: 0.0062959
time: 18.62 seconds

## Relational analysis of NS_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062252, upper bound: 0.0062984
time: 8.28 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1869224, 0.4344618, -0.1871343, 0.4345948, -0.3199014, 0.3195664
1: -1.4440038, -0.2961225, -1.4450725, -0.2958775, -0.5858177, 0.5851609
2: -3.2345347, -2.2579496, -3.2350562, -2.2533598, -0.1235368, 0.1136029
3: -4.1961527, -2.7095251, -4.1971855, -2.7056744, -0.5055538, 0.5004472
4: -2.8553076, -1.4613159, -2.8561256, -1.4560592, -0.2437511, 0.2345946
5: -5.2417283, -3.6414387, -5.2429123, -3.6376963, -0.4628592, 0.4578025
6: -5.8263254, -4.1405492, -5.8264399, -4.1344180, -0.3304985, 0.3198874
7: -2.7996104, -1.2560704, -2.8003812, -1.2538801, -0.4448501, 0.4361131
8: 0.9780055, 1.1545702, 0.9772428, 1.1546804, -0.0161370, 0.0195940
9: -0.0996909, 0.3799010, -0.0996778, 0.3799095, -0.0617645, 0.0614613

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 398
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 2197
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 3081
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 388
type: B, layer: 1, pos: 388
type: B, layer: 1, pos: 2576
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3475
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 373
type: A, layer: 1, pos: 2107
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: A, layer: 1, pos: 2366
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: A, layer: 1, pos: 2799
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: A, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 3489
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 3076
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2830
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2386
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: B, layer: 1, pos: 2575
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3042
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2842

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 398

## Relational analysis of NS_A2_A1_B2_A1

### Relational analysis result of NS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062961, upper bound: 0.0061809
time: 35.29 seconds

## Relational analysis of NS_A2_A1_B2_A2

### Relational analysis result of NS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062964, upper bound: 0.0062392
time: 11.88 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1875337, 0.4344293, -0.1875249, 0.4344139, -0.3193174, 0.3201719
1: -1.4450784, -0.2953655, -1.4450756, -0.2953627, -0.5871463, 0.5848598
2: -3.2365794, -2.2567909, -3.2363653, -2.2567897, -0.1168438, 0.1190881
3: -4.1994114, -2.7083559, -4.1992865, -2.7083077, -0.5010393, 0.5045725
4: -2.8590112, -1.4602907, -2.8587940, -1.4602907, -0.2347693, 0.2442357
5: -5.2456846, -3.6399994, -5.2455683, -3.6399410, -0.4587807, 0.4631336
6: -5.8238225, -4.1402979, -5.8234310, -4.1402569, -0.3216679, 0.3209170
7: -2.8051314, -1.2545303, -2.8050814, -1.2545294, -0.4395777, 0.4421334
8: 0.9777474, 1.1551588, 0.9777470, 1.1551350, -0.0172758, 0.0184177
9: -0.0997211, 0.3801531, -0.0997211, 0.3801486, -0.0619470, 0.0610020

Time for backsubstitution: 5.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 398
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 2197
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3081
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 373
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 403
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 400
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: A, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: B, layer: 1, pos: 2830
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3042
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 398

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062092, upper bound: 0.0063019
time: 3.57 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062767, upper bound: 0.0062967
time: 3.30 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1876933, 0.4345942, -0.1878146, 0.4345954, -0.3196135, 0.3205791
1: -1.4450403, -0.2953651, -1.4451257, -0.2952087, -0.5873080, 0.5850369
2: -3.2385943, -2.2567911, -3.2386425, -2.2533596, -0.1220755, 0.1192788
3: -4.2005358, -2.7083402, -4.2010536, -2.7056701, -0.5046672, 0.5058296
4: -2.8615189, -1.4602907, -2.8616097, -1.4560580, -0.2411771, 0.2444471
5: -5.2467060, -3.6399984, -5.2473049, -3.6376951, -0.4618391, 0.4641490
6: -5.8272772, -4.1402969, -5.8272805, -4.1344175, -0.3302048, 0.3211644
7: -2.8047810, -1.2545301, -2.8049288, -1.2538786, -0.4436569, 0.4425558
8: 0.9777473, 1.1554352, 0.9772428, 1.1554435, -0.0173006, 0.0192878
9: -0.0997212, 0.3800315, -0.0996822, 0.3800257, -0.0619517, 0.0614349

Time for backsubstitution: 5.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 398
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 2197
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 388
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3475
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: A, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 3076
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2386
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 352
type: A, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2095
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 398

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062299, upper bound: 0.0062976
time: 9.55 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062977, upper bound: 0.0062974
time: 6.42 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 22.01 seconds
NS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 8, lower bound: -0.0061385, upper bound: 0.0062994
NS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 8, lower bound: -0.0062059, upper bound: 0.0062975
NS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 8, lower bound: -0.0061583, upper bound: 0.0062959
NS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 8, lower bound: -0.0062252, upper bound: 0.0062984
NS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 8, lower bound: -0.0062961, upper bound: 0.0061809
NS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 8, lower bound: -0.0062964, upper bound: 0.0062392
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 8, lower bound: -0.0062092, upper bound: 0.0063019
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 8, lower bound: -0.0062767, upper bound: 0.0062967
NS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 8, lower bound: -0.0062299, upper bound: 0.0062976
NS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 22.01
Output dim: 8, lower bound: -0.0062977, upper bound: 0.0062974

## BFS NS instance: NS_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.1864738, 0.4341819, -0.1862754, 0.4341958, -0.3179825, 0.3188504
1: -1.4449949, -0.2972888, -1.4450731, -0.2972793, -0.5851120, 0.5841467
2: -3.2359385, -2.2599545, -3.2363544, -2.2595286, -0.1125084, 0.1186411
3: -4.2006993, -2.7087779, -4.1992140, -2.7087049, -0.4987377, 0.5036777
4: -2.8582478, -1.4610900, -2.8580475, -1.4609916, -0.2333429, 0.2436554
5: -5.2471972, -3.6404471, -5.2454758, -3.6403518, -0.4565812, 0.4620273
6: -5.8248005, -4.1407785, -5.8234310, -4.1407156, -0.3184178, 0.3198159
7: -2.8050959, -1.2580382, -2.8050642, -1.2578422, -0.4355434, 0.4416905
8: 0.9796935, 1.1544728, 0.9794139, 1.1548578, -0.0174057, 0.0160507
9: -0.0997032, 0.3792670, -0.0997209, 0.3792445, -0.0608035, 0.0607478

Time for backsubstitution: 5.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2197
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 373
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 403
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: A, layer: 1, pos: 400
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: A, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 352
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3042
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2197

## Relational analysis of NS_A1_A2_B1_B1_A1

### Relational analysis result of NS_A1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0061372, upper bound: 0.0062386
time: 10.76 seconds

## Relational analysis of NS_A1_A2_B1_B1_A2

### Relational analysis result of NS_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061366, upper bound: 0.0062967
time: 3.75 seconds

## BFS NS instance: NS_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.1874924, 0.4341820, -0.1876913, 0.4360712, -0.3209894, 0.3201106
1: -1.4449966, -0.2966136, -1.4476478, -0.2963567, -0.5857704, 0.5874764
2: -3.2359464, -2.2606807, -3.2362769, -2.2591329, -0.1132634, 0.1214669
3: -4.2007556, -2.7108951, -4.2027597, -2.7098427, -0.4996766, 0.5133413
4: -2.8588650, -1.4610896, -2.8588116, -1.4588587, -0.2361900, 0.2441548
5: -5.2472687, -3.6427069, -5.2496529, -3.6414711, -0.4573778, 0.4734970
6: -5.8248005, -4.1419387, -5.8256168, -4.1413565, -0.3197276, 0.3253114
7: -2.8050213, -1.2611825, -2.8038013, -1.2591530, -0.4359057, 0.4444705
8: 0.9796946, 1.1547031, 0.9785895, 1.1551833, -0.0176528, 0.0171189
9: -0.0997033, 0.3796177, -0.1009681, 0.3796980, -0.0612267, 0.0624050

Time for backsubstitution: 5.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2197
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 398
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: B, layer: 1, pos: 2830
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 352
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 2842
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2361

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2197

## Relational analysis of NS_A1_A2_B1_B2_A1

### Relational analysis result of NS_A1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062037, upper bound: 0.0062382
time: 4.10 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2

### Relational analysis result of NS_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062050, upper bound: 0.0062946
time: 7.30 seconds

## BFS NS instance: NS_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.1866338, 0.4343466, -0.1865654, 0.4343773, -0.3182785, 0.3192580
1: -1.4449564, -0.2972881, -1.4451232, -0.2971253, -0.5852740, 0.5843243
2: -3.2379541, -2.2599545, -3.2386315, -2.2560987, -0.1177401, 0.1188315
3: -4.2018242, -2.7087622, -4.2009802, -2.7060666, -0.5023625, 0.5049321
4: -2.8607543, -1.4610901, -2.8608637, -1.4567590, -0.2397508, 0.2438670
5: -5.2482185, -3.6404464, -5.2472105, -3.6381054, -0.4596362, 0.4630385
6: -5.8282557, -4.1407776, -5.8272800, -4.1348772, -0.3269545, 0.3200632
7: -2.8047452, -1.2580380, -2.8049116, -1.2571914, -0.4396225, 0.4421131
8: 0.9796935, 1.1547492, 0.9789099, 1.1551661, -0.0174305, 0.0169208
9: -0.0997033, 0.3791454, -0.0996819, 0.3791216, -0.0608082, 0.0611807

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2197
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 388
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: A, layer: 1, pos: 400
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 3076
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2830
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2386
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 352
type: A, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2095
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2361

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2197

## Relational analysis of NS_A1_A2_B2_B1_A1

### Relational analysis result of NS_A1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0061565, upper bound: 0.0062365
time: 262.39 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2

### Relational analysis result of NS_A1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0061561, upper bound: 0.0062932
time: 4.44 seconds

## BFS NS instance: NS_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.1876524, 0.4343468, -0.1879816, 0.4362525, -0.3212855, 0.3205183
1: -1.4449583, -0.2966131, -1.4476978, -0.2962030, -0.5859319, 0.5876539
2: -3.2379618, -2.2606807, -3.2385535, -2.2557025, -0.1184951, 0.1216571
3: -4.2018800, -2.7108798, -4.2045026, -2.7072048, -0.5033041, 0.5146032
4: -2.8613715, -1.4610894, -2.8616271, -1.4546267, -0.2425978, 0.2443666
5: -5.2482905, -3.6427054, -5.2513628, -3.6392245, -0.4604344, 0.4745087
6: -5.8282557, -4.1419382, -5.8294659, -4.1355176, -0.3282644, 0.3255588
7: -2.8046710, -1.2611821, -2.8036530, -1.2585027, -0.4399853, 0.4448930
8: 0.9796946, 1.1549795, 0.9780854, 1.1554916, -0.0176776, 0.0179890
9: -0.0997034, 0.3794961, -0.1009294, 0.3795752, -0.0612313, 0.0628379

Time for backsubstitution: 5.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2197
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 398
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 3076
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: B, layer: 1, pos: 2830
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 352
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 352
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 2842
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2361

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2197

## Relational analysis of NS_A1_A2_B2_B2_A1

### Relational analysis result of NS_A1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062244, upper bound: 0.0062363
time: 30.96 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062233, upper bound: 0.0062954
time: 3.55 seconds

## BFS NS instance: NS_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1857089, 0.4344616, -0.1860726, 0.4345945, -0.3186505, 0.3184701
1: -1.4440013, -0.2969358, -1.4450705, -0.2965908, -0.5850806, 0.5843206
2: -3.2345247, -2.2579830, -3.2350466, -2.2533879, -0.1225584, 0.1125451
3: -4.1960773, -2.7096229, -4.1971159, -2.7057571, -0.5047629, 0.4995462
4: -2.8545628, -1.4613166, -2.8554742, -1.4560597, -0.2430325, 0.2339564
5: -5.2416320, -3.6415014, -5.2428226, -3.6377475, -0.4621273, 0.4569362
6: -5.8263235, -4.1407671, -5.8264380, -4.1346140, -0.3291043, 0.3182721
7: -2.7995930, -1.2561296, -2.8003650, -1.2539268, -0.4442415, 0.4355426
8: 0.9780059, 1.1542931, 0.9772431, 1.1544372, -0.0158905, 0.0193307
9: -0.0996906, 0.3794940, -0.0996776, 0.3795408, -0.0613812, 0.0610239

Time for backsubstitution: 6.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2197
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 3081
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 388
type: B, layer: 1, pos: 388
type: B, layer: 1, pos: 2576
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3475
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 373
type: A, layer: 1, pos: 2107
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 403
type: B, layer: 1, pos: 2374
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: B, layer: 1, pos: 2809
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 2366
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: A, layer: 1, pos: 2799
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 2824
type: B, layer: 1, pos: 2824
type: A, layer: 1, pos: 2373
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 398
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 3489
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 3076
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2830
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2386
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: B, layer: 1, pos: 2362
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 3032
type: A, layer: 1, pos: 3032
type: B, layer: 1, pos: 2575
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3042
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2842

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2197

## Relational analysis of NS_A2_A1_B2_A1_B1

### Relational analysis result of NS_A2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062353, upper bound: 0.0061677
time: 8.24 seconds

## Relational analysis of NS_A2_A1_B2_A1_B2

### Relational analysis result of NS_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062939, upper bound: 0.0061737
time: 3.02 seconds

## BFS NS instance: NS_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1871261, 0.4363369, -0.1870908, 0.4345944, -0.3199075, 0.3214802
1: -1.4465778, -0.2960110, -1.4450727, -0.2959157, -0.5884078, 0.5849758
2: -3.2344542, -2.2575822, -3.2350535, -2.2541149, -0.1255148, 0.1133005
3: -4.1997805, -2.7107606, -4.1971722, -2.7078788, -0.5146766, 0.5004858
4: -2.8553262, -1.4591815, -2.8560913, -1.4560592, -0.2435963, 0.2368045
5: -5.2459903, -3.6426222, -5.2428951, -3.6400075, -0.4738763, 0.4577327
6: -5.8285613, -4.1414013, -5.8264389, -4.1357746, -0.3350372, 0.3195829
7: -2.7983308, -1.2574419, -2.8002911, -1.2570729, -0.4470705, 0.4359097
8: 0.9771816, 1.1546191, 0.9772441, 1.1546674, -0.0169639, 0.0195764
9: -0.1009345, 0.3799345, -0.0996777, 0.3798851, -0.0630386, 0.0614435

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2197
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 3081
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 3080
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 388
type: B, layer: 1, pos: 388
type: B, layer: 1, pos: 2576
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2166
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 3475
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 373
type: A, layer: 1, pos: 2107
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 403
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 2374
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 3082
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 2809
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 3033
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 400
type: B, layer: 1, pos: 3276
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2366
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 335
type: B, layer: 1, pos: 3048
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 2799
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2373
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 3489
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 3076
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 2117
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2830
type: A, layer: 1, pos: 2830
type: B, layer: 1, pos: 2386
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 2362
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3274
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 352
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 2575
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 2095
type: A, layer: 1, pos: 2842
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2361

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2197

## Relational analysis of NS_A2_A1_B2_A2_B1

### Relational analysis result of NS_A2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062358, upper bound: 0.0062283
time: 21.95 seconds

## Relational analysis of NS_A2_A1_B2_A2_B2

### Relational analysis result of NS_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062942, upper bound: 0.0062304
time: 3.33 seconds

## BFS NS instance: NS_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.1864717, 0.4344290, -0.1863113, 0.4344136, -0.3182225, 0.3189200
1: -1.4450765, -0.2960787, -1.4450734, -0.2961758, -0.5863109, 0.5841230
2: -3.2365699, -2.2568188, -3.2363544, -2.2568254, -0.1157853, 0.1181520
3: -4.1993423, -2.7084398, -4.1992140, -2.7084064, -0.5001358, 0.5037878
4: -2.8583596, -1.4602911, -2.8580487, -1.4602909, -0.2341299, 0.2435800
5: -5.2455959, -3.6400509, -5.2454758, -3.6400042, -0.4579120, 0.4624075
6: -5.8238211, -4.1404934, -5.8234310, -4.1404681, -0.3200530, 0.3195232
7: -2.8051167, -1.2545772, -2.8050647, -1.2545887, -0.4390040, 0.4415248
8: 0.9777477, 1.1549155, 0.9777473, 1.1548578, -0.0170158, 0.0181743
9: -0.0997210, 0.3797804, -0.0997210, 0.3797292, -0.0615090, 0.0606188

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2197
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 373
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 403
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: A, layer: 1, pos: 400
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: A, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 352
type: B, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3042
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2197

## Relational analysis of NS_A2_A2_B1_B1_A1

### Relational analysis result of NS_A2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062052, upper bound: 0.0062386
time: 3.17 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2

### Relational analysis result of NS_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062078, upper bound: 0.0062936
time: 8.30 seconds

## BFS NS instance: NS_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.1874903, 0.4344293, -0.1877275, 0.4362889, -0.3212295, 0.3201804
1: -1.4450786, -0.2954036, -1.4476482, -0.2952532, -0.5869694, 0.5874524
2: -3.2365770, -2.2575445, -3.2362781, -2.2564299, -0.1165404, 0.1209808
3: -4.1993980, -2.7105570, -4.2027602, -2.7095437, -0.5010763, 0.5134851
4: -2.8589771, -1.4602908, -2.8588121, -1.4581581, -0.2369769, 0.2440793
5: -5.2456679, -3.6423101, -5.2496538, -3.6411235, -0.4587104, 0.4739170
6: -5.8238215, -4.1416540, -5.8256168, -4.1411085, -0.3213628, 0.3250190
7: -2.8050423, -1.2577212, -2.8038018, -1.2558997, -0.4393665, 0.4443049
8: 0.9777488, 1.1551459, 0.9769229, 1.1551834, -0.0172630, 0.0192424
9: -0.0997211, 0.3801312, -0.1009682, 0.3801829, -0.0619320, 0.0622760

Time for backsubstitution: 6.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2197
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 398
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: B, layer: 1, pos: 2830
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 352
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 2842
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2361

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2197

## Relational analysis of NS_A2_A2_B1_B2_A1

### Relational analysis result of NS_A2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062754, upper bound: 0.0062395
time: 170.50 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2

### Relational analysis result of NS_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062742, upper bound: 0.0062961
time: 12.24 seconds

## BFS NS instance: NS_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.1866315, 0.4345938, -0.1866012, 0.4345950, -0.3185186, 0.3193277
1: -1.4450384, -0.2960782, -1.4451232, -0.2960220, -0.5864726, 0.5843002
2: -3.2385857, -2.2568190, -3.2386322, -2.2533956, -0.1210170, 0.1183426
3: -4.2004666, -2.7084241, -4.2009802, -2.7057676, -0.5037615, 0.5050457
4: -2.8608673, -1.4602911, -2.8608644, -1.4560581, -0.2405377, 0.2437914
5: -5.2466173, -3.6400504, -5.2472115, -3.6377587, -0.4609683, 0.4634228
6: -5.8272767, -4.1404929, -5.8272800, -4.1346288, -0.3285897, 0.3197705
7: -2.8047655, -1.2545768, -2.8049114, -1.2539380, -0.4430833, 0.4419473
8: 0.9777477, 1.1551918, 0.9772431, 1.1551664, -0.0170407, 0.0190443
9: -0.0997212, 0.3796588, -0.0996819, 0.3796062, -0.0615137, 0.0610517

Time for backsubstitution: 6.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2197
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 388
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3475
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: A, layer: 1, pos: 400
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: A, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 3076
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 2830
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2386
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 352
type: A, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2095
type: A, layer: 1, pos: 3042
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 2095
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2361

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2197

## Relational analysis of NS_A2_A2_B2_B1_A1

### Relational analysis result of NS_A2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062267, upper bound: 0.0062390
time: 10.02 seconds

## Relational analysis of NS_A2_A2_B2_B1_A2

### Relational analysis result of NS_A2_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062253, upper bound: 0.0062913
time: 7.85 seconds

## BFS NS instance: NS_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.1876500, 0.4345941, -0.1880176, 0.4364702, -0.3215256, 0.3205881
1: -1.4450403, -0.2954031, -1.4476976, -0.2950994, -0.5871313, 0.5876296
2: -3.2385929, -2.2575448, -3.2385542, -2.2529998, -0.1217721, 0.1211713
3: -4.2005229, -2.7105417, -4.2045031, -2.7069068, -0.5047047, 0.5147504
4: -2.8614841, -1.4602904, -2.8616281, -1.4539261, -0.2433846, 0.2442910
5: -5.2466888, -3.6423087, -5.2513618, -3.6388774, -0.4617685, 0.4749324
6: -5.8272753, -4.1416531, -5.8294668, -4.1352701, -0.3298995, 0.3252664
7: -2.8046913, -1.2577214, -2.8036532, -1.2552490, -0.4434459, 0.4447271
8: 0.9777488, 1.1554222, 0.9764187, 1.1554917, -0.0172879, 0.0201125
9: -0.0997212, 0.3800094, -0.1009294, 0.3800602, -0.0619367, 0.0627089

Time for backsubstitution: 6.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2197
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3475
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 398
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 3076
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: B, layer: 1, pos: 2830
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 352
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 352
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 2575
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 2842
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2361

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 2197

## Relational analysis of NS_A2_A2_B2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062933, upper bound: 0.0062391
time: 3.44 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062937, upper bound: 0.0062916
time: 8.22 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 17.80 seconds
NS_A1_A2_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 17.80
Output dim: 8, lower bound: -0.0061372, upper bound: 0.0062386
NS_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.80
Output dim: 8, lower bound: -0.0061366, upper bound: 0.0062967
NS_A1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 17.80
Output dim: 8, lower bound: -0.0062037, upper bound: 0.0062382
NS_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.80
Output dim: 8, lower bound: -0.0062050, upper bound: 0.0062946
NS_A1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 17.80
Output dim: 8, lower bound: -0.0061565, upper bound: 0.0062365
NS_A1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 5, time: 17.80
Output dim: 8, lower bound: -0.0061561, upper bound: 0.0062932
NS_A1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 17.80
Output dim: 8, lower bound: -0.0062244, upper bound: 0.0062363
NS_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.80
Output dim: 8, lower bound: -0.0062233, upper bound: 0.0062954
NS_A2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 17.80
Output dim: 8, lower bound: -0.0062353, upper bound: 0.0061677
NS_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 17.80
Output dim: 8, lower bound: -0.0062939, upper bound: 0.0061737
NS_A2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 17.80
Output dim: 8, lower bound: -0.0062358, upper bound: 0.0062283
NS_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 17.80
Output dim: 8, lower bound: -0.0062942, upper bound: 0.0062304
NS_A2_A2_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 17.80
Output dim: 8, lower bound: -0.0062052, upper bound: 0.0062386
NS_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 17.80
Output dim: 8, lower bound: -0.0062078, upper bound: 0.0062936
NS_A2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 5, time: 17.80
Output dim: 8, lower bound: -0.0062754, upper bound: 0.0062395
NS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.80
Output dim: 8, lower bound: -0.0062742, upper bound: 0.0062961
NS_A2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 17.80
Output dim: 8, lower bound: -0.0062267, upper bound: 0.0062390
NS_A2_A2_B2_B1_A2, status: Status.VERIFIED, split count: 5, time: 17.80
Output dim: 8, lower bound: -0.0062253, upper bound: 0.0062913
NS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 17.80
Output dim: 8, lower bound: -0.0062933, upper bound: 0.0062391
NS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 17.80
Output dim: 8, lower bound: -0.0062937, upper bound: 0.0062916

## BFS NS instance: NS_A1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1864251, 0.4341817, -0.1862332, 0.4341959, -0.3170595, 0.3188269
1: -1.4449930, -0.2973047, -1.4450719, -0.2972929, -0.5851112, 0.5833632
2: -3.2358675, -2.2599547, -3.2362921, -2.2595286, -0.1079532, 0.1186399
3: -4.2006273, -2.7087793, -4.1991520, -2.7087064, -0.4950291, 0.5036203
4: -2.8579195, -1.4610902, -2.8577633, -1.4609914, -0.2243257, 0.2436469
5: -5.2471161, -3.6404476, -5.2454042, -3.6403522, -0.4521333, 0.4620032
6: -5.8247595, -4.1407785, -5.8233957, -4.1407166, -0.3170516, 0.3198145
7: -2.8050003, -1.2580379, -2.8049817, -1.2578422, -0.4308266, 0.4416584
8: 0.9796935, 1.1544535, 0.9794139, 1.1548408, -0.0174048, 0.0149754
9: -0.0997029, 0.3792576, -0.0997208, 0.3792360, -0.0608025, 0.0605650

Time for backsubstitution: 6.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 373
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 403
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: A, layer: 1, pos: 400
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 398
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 2117
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2842
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2361

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 158

## Relational analysis of NS_A1_A2_B1_B1_A2_B1

### Relational analysis result of NS_A1_A2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0061164, upper bound: 0.0062922
time: 68.35 seconds

## Relational analysis of NS_A1_A2_B1_B1_A2_B2

### Relational analysis result of NS_A1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061361, upper bound: 0.0062959
time: 19.85 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1874440, 0.4341819, -0.1876493, 0.4360710, -0.3200675, 0.3200871
1: -1.4449948, -0.2966294, -1.4476464, -0.2963707, -0.5857689, 0.5866982
2: -3.2358747, -2.2606807, -3.2362144, -2.2591326, -0.1087081, 0.1214509
3: -4.2006841, -2.7108972, -4.2026873, -2.7098441, -0.4960670, 0.5132611
4: -2.8585372, -1.4610896, -2.8585272, -1.4588587, -0.2271853, 0.2441379
5: -5.2471876, -3.6427069, -5.2495704, -3.6414714, -0.4531347, 0.4734471
6: -5.8247585, -4.1419392, -5.8255787, -4.1413574, -0.3186150, 0.3252978
7: -2.8049259, -1.2611824, -2.8037186, -1.2591534, -0.4312498, 0.4444267
8: 0.9796946, 1.1546838, 0.9785895, 1.1551664, -0.0176510, 0.0160479
9: -0.0997031, 0.3796083, -0.1009680, 0.3796901, -0.0612254, 0.0622247

Time for backsubstitution: 6.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 373
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 398
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: B, layer: 1, pos: 2830
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 352
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 2575
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2361

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 158

## Relational analysis of NS_A1_A2_B1_B2_A2_B1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061836, upper bound: 0.0062969
time: 3.41 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062029, upper bound: 0.0062973
time: 5.31 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1876035, 0.4343468, -0.1879392, 0.4362525, -0.3203637, 0.3204948
1: -1.4449568, -0.2966289, -1.4476962, -0.2962169, -0.5859301, 0.5868754
2: -3.2378900, -2.2606807, -3.2384906, -2.2557027, -0.1139398, 0.1216411
3: -4.2018085, -2.7108815, -4.2044296, -2.7072067, -0.4996942, 0.5145229
4: -2.8610437, -1.4610898, -2.8613431, -1.4546269, -0.2335931, 0.2443498
5: -5.2482095, -3.6427054, -5.2512784, -3.6392250, -0.4561915, 0.4744587
6: -5.8282146, -4.1419382, -5.8294287, -4.1355186, -0.3271517, 0.3255452
7: -2.8045750, -1.2611823, -2.8035700, -1.2585030, -0.4353293, 0.4448491
8: 0.9796947, 1.1549600, 0.9780854, 1.1554747, -0.0176759, 0.0169179
9: -0.0997033, 0.3794868, -0.1009292, 0.3795670, -0.0612301, 0.0626576

Time for backsubstitution: 6.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 398
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: B, layer: 1, pos: 2830
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 352
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2361

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 158

## Relational analysis of NS_A1_A2_B2_B2_A2_B1

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062044, upper bound: 0.0062950
time: 3.22 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_B2

### Relational analysis result of NS_A1_A2_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062234, upper bound: 0.0062925
time: 8.15 seconds

## BFS NS instance: NS_A2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1856667, 0.4344615, -0.1860239, 0.4345944, -0.3186170, 0.3175475
1: -1.4439998, -0.2969493, -1.4450686, -0.2966066, -0.5842957, 0.5843193
2: -3.2344069, -2.2579830, -3.2349100, -2.2533879, -0.1225568, 0.1079774
3: -4.1960144, -2.7096243, -4.1970439, -2.7057590, -0.5046946, 0.4957111
4: -2.8542790, -1.4613166, -2.8551469, -1.4560599, -0.2430214, 0.2249217
5: -5.2415619, -3.6415024, -5.2427406, -3.6377478, -0.4621027, 0.4524646
6: -5.8262873, -4.1407671, -5.8263979, -4.1346149, -0.3291029, 0.3169044
7: -2.7995105, -1.2561295, -2.8002691, -1.2539269, -0.4441995, 0.4308278
8: 0.9780059, 1.1542760, 0.9772431, 1.1544178, -0.0148090, 0.0193295
9: -0.0996906, 0.3794796, -0.0996774, 0.3795288, -0.0611966, 0.0610224

Time for backsubstitution: 6.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 3081
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 3080
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 388
type: B, layer: 1, pos: 388
type: B, layer: 1, pos: 2576
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2166
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 3475
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 373
type: A, layer: 1, pos: 2107
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 403
type: B, layer: 1, pos: 2374
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 3082
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 2809
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 3033
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 2366
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 335
type: B, layer: 1, pos: 3048
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 2799
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 2824
type: B, layer: 1, pos: 2824
type: A, layer: 1, pos: 2373
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 3489
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 3076
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 2117
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2386
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: B, layer: 1, pos: 2362
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3274
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 2575
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3042
type: B, layer: 1, pos: 2095
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2842

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 158

## Relational analysis of NS_A2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062935, upper bound: 0.0061579
time: 88.35 seconds

## Relational analysis of NS_A2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062948, upper bound: 0.0061651
time: 26.61 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1870838, 0.4363370, -0.1870421, 0.4345945, -0.3198736, 0.3205587
1: -1.4465759, -0.2960249, -1.4450707, -0.2959313, -0.5876287, 0.5849739
2: -3.2343361, -2.2575824, -3.2349176, -2.2541146, -0.1254975, 0.1087349
3: -4.1997070, -2.7107620, -4.1971006, -2.7078805, -0.5145968, 0.4967494
4: -2.8550420, -1.4591814, -2.8557644, -1.4560595, -0.2435791, 0.2277849
5: -5.2459078, -3.6426222, -5.2428145, -3.6400080, -0.4738266, 0.4534660
6: -5.8284912, -4.1414013, -5.8263979, -4.1357756, -0.3349706, 0.3184700
7: -2.7982478, -1.2574420, -2.8001957, -1.2570724, -0.4470165, 0.4313448
8: 0.9771816, 1.1546023, 0.9772441, 1.1546481, -0.0158912, 0.0195746
9: -0.1009343, 0.3799264, -0.0996775, 0.3798758, -0.0628565, 0.0614419

Time for backsubstitution: 6.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 3081
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 3080
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 388
type: B, layer: 1, pos: 388
type: B, layer: 1, pos: 2576
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2166
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 3475
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 373
type: A, layer: 1, pos: 2107
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 398
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 403
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 2374
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 3082
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 2809
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 3033
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 400
type: B, layer: 1, pos: 3276
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2366
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 335
type: B, layer: 1, pos: 3048
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 2799
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2373
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 3489
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 3076
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 2117
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2830
type: A, layer: 1, pos: 2830
type: B, layer: 1, pos: 2386
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 2362
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3274
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 352
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 2575
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 2842
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2361

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 158

## Relational analysis of NS_A2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062928, upper bound: 0.0062229
time: 3.59 seconds

## Relational analysis of NS_A2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062937, upper bound: 0.0062236
time: 14.11 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1864230, 0.4344290, -0.1862690, 0.4344136, -0.3172996, 0.3188966
1: -1.4450750, -0.2960945, -1.4450719, -0.2961895, -0.5863102, 0.5833390
2: -3.2364981, -2.2568192, -3.2362928, -2.2568259, -0.1112301, 0.1181507
3: -4.1992707, -2.7084415, -4.1991520, -2.7084076, -0.4964274, 0.5037303
4: -2.8580322, -1.4602914, -2.8577652, -1.4602911, -0.2251126, 0.2435715
5: -5.2455149, -3.6400514, -5.2454057, -3.6400046, -0.4534639, 0.4623833
6: -5.8237810, -4.1404924, -5.8233953, -4.1404676, -0.3186868, 0.3195217
7: -2.8050210, -1.2545769, -2.8049817, -1.2545886, -0.4342874, 0.4414928
8: 0.9777477, 1.1548960, 0.9777473, 1.1548409, -0.0170150, 0.0170989
9: -0.0997208, 0.3797708, -0.0997207, 0.3797207, -0.0615079, 0.0604360

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 373
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 403
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: A, layer: 1, pos: 400
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2842
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2361

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 158

## Relational analysis of NS_A2_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061867, upper bound: 0.0062973
time: 3.70 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062066, upper bound: 0.0062946
time: 17.62 seconds

## BFS NS instance: NS_A2_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1874416, 0.4344293, -0.1876852, 0.4362888, -0.3203075, 0.3201569
1: -1.4450768, -0.2954195, -1.4476466, -0.2952672, -0.5869677, 0.5866739
2: -3.2365062, -2.2575450, -3.2362151, -2.2564299, -0.1119851, 0.1209648
3: -4.1993270, -2.7105589, -4.2026868, -2.7095456, -0.4974670, 0.5134048
4: -2.8586497, -1.4602909, -2.8585281, -1.4581585, -0.2279722, 0.2440625
5: -5.2455864, -3.6423106, -5.2495704, -3.6411242, -0.4544672, 0.4738669
6: -5.8237810, -4.1416540, -5.8255796, -4.1411095, -0.3202503, 0.3250054
7: -2.8049469, -1.2577214, -2.8037183, -1.2558999, -0.4347104, 0.4442611
8: 0.9777488, 1.1551263, 0.9769229, 1.1551665, -0.0172613, 0.0181714
9: -0.0997210, 0.3801221, -0.1009680, 0.3801748, -0.0619308, 0.0620957

Time for backsubstitution: 6.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 373
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 398
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: B, layer: 1, pos: 2830
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 352
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 2575
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2361

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 158

## Relational analysis of NS_A2_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062561, upper bound: 0.0062962
time: 3.46 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2_B2

### Relational analysis result of NS_A2_A2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062757, upper bound: 0.0062932
time: 34.26 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1869648, 0.4345151, -0.1874083, 0.4364699, -0.3208344, 0.3196970
1: -1.4443586, -0.2960477, -1.4476579, -0.2956636, -0.5859413, 0.5869598
2: -3.2354460, -2.2582281, -3.2357910, -2.2530005, -0.1182305, 0.1170357
3: -4.1974010, -2.7112885, -4.2018046, -2.7069101, -0.5015678, 0.5109036
4: -2.8553743, -1.4613428, -2.8562570, -1.4539266, -0.2363293, 0.2358273
5: -5.2431355, -3.6431479, -5.2482901, -3.6388783, -0.4580543, 0.4706495
6: -5.8263178, -4.1418223, -5.8286343, -4.1352701, -0.3288291, 0.3242411
7: -2.8007617, -1.2586799, -2.8002191, -1.2552497, -0.4394119, 0.4400607
8: 0.9779416, 1.1546308, 0.9764187, 1.1547992, -0.0162754, 0.0192747
9: -0.0997013, 0.3798715, -0.1009259, 0.3799397, -0.0617479, 0.0625495

Time for backsubstitution: 6.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3475
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 398
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 3076
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: B, layer: 1, pos: 2830
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 352
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 352
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 2575
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 2842
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2361

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 158

## Relational analysis of NS_A2_A2_B2_B2_A1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062743, upper bound: 0.0062366
time: 40.76 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062929, upper bound: 0.0062352
time: 8.84 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1876015, 0.4345942, -0.1879754, 0.4364702, -0.3206038, 0.3205646
1: -1.4450386, -0.2954189, -1.4476964, -0.2951133, -0.5871295, 0.5868515
2: -3.2385216, -2.2575448, -3.2384903, -2.2529998, -0.1172168, 0.1211552
3: -4.2004514, -2.7105432, -4.2044296, -2.7069082, -0.5010948, 0.5146700
4: -2.8611569, -1.4602908, -2.8613441, -1.4539263, -0.2343800, 0.2442742
5: -5.2466078, -3.6423085, -5.2512794, -3.6388774, -0.4575256, 0.4748822
6: -5.8272352, -4.1416531, -5.8294291, -4.1352701, -0.3287870, 0.3252527
7: -2.8045952, -1.2577211, -2.8035705, -1.2552493, -0.4387899, 0.4446832
8: 0.9777488, 1.1554027, 0.9764186, 1.1554749, -0.0172861, 0.0190415
9: -0.0997212, 0.3800002, -0.1009292, 0.3800520, -0.0619355, 0.0625286

Time for backsubstitution: 6.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3475
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 398
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: B, layer: 1, pos: 2830
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 352
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2361

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 158

## Relational analysis of NS_A2_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062760, upper bound: 0.0062910
time: 15.28 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062937, upper bound: 0.0062952
time: 19.89 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 41.29 seconds
NS_A1_A2_B1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 41.29
Output dim: 8, lower bound: -0.0061164, upper bound: 0.0062922
NS_A1_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 41.29
Output dim: 8, lower bound: -0.0061361, upper bound: 0.0062959
NS_A1_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 41.29
Output dim: 8, lower bound: -0.0061836, upper bound: 0.0062969
NS_A1_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 41.29
Output dim: 8, lower bound: -0.0062029, upper bound: 0.0062973
NS_A1_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 41.29
Output dim: 8, lower bound: -0.0062044, upper bound: 0.0062950
NS_A1_A2_B2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 41.29
Output dim: 8, lower bound: -0.0062234, upper bound: 0.0062925
NS_A2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 41.29
Output dim: 8, lower bound: -0.0062935, upper bound: 0.0061579
NS_A2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 41.29
Output dim: 8, lower bound: -0.0062948, upper bound: 0.0061651
NS_A2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 41.29
Output dim: 8, lower bound: -0.0062928, upper bound: 0.0062229
NS_A2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 41.29
Output dim: 8, lower bound: -0.0062937, upper bound: 0.0062236
NS_A2_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 41.29
Output dim: 8, lower bound: -0.0061867, upper bound: 0.0062973
NS_A2_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 41.29
Output dim: 8, lower bound: -0.0062066, upper bound: 0.0062946
NS_A2_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 41.29
Output dim: 8, lower bound: -0.0062561, upper bound: 0.0062962
NS_A2_A2_B1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 41.29
Output dim: 8, lower bound: -0.0062757, upper bound: 0.0062932
NS_A2_A2_B2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 41.29
Output dim: 8, lower bound: -0.0062743, upper bound: 0.0062366
NS_A2_A2_B2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 41.29
Output dim: 8, lower bound: -0.0062929, upper bound: 0.0062352
NS_A2_A2_B2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 41.29
Output dim: 8, lower bound: -0.0062760, upper bound: 0.0062910
NS_A2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 41.29
Output dim: 8, lower bound: -0.0062937, upper bound: 0.0062952

## BFS NS instance: NS_A1_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1864229, 0.4341817, -0.1862307, 0.4341959, -0.3170533, 0.3180402
1: -1.4449931, -0.2973102, -1.4450717, -0.2972993, -0.5843350, 0.5833625
2: -3.2358491, -2.2599545, -3.2362721, -2.2595284, -0.1079477, 0.1150523
3: -4.2005954, -2.7087798, -4.1991158, -2.7087064, -0.4949777, 0.4981956
4: -2.8579097, -1.4610901, -2.8577533, -1.4609916, -0.2243160, 0.2364102
5: -5.2470713, -3.6404476, -5.2453542, -3.6403522, -0.4521104, 0.4565418
6: -5.8247476, -4.1407785, -5.8233833, -4.1407166, -0.3170412, 0.3184017
7: -2.8049908, -1.2580380, -2.8049712, -1.2578422, -0.4307900, 0.4358331
8: 0.9796935, 1.1544492, 0.9794139, 1.1548361, -0.0168203, 0.0149728
9: -0.0997028, 0.3792574, -0.0997207, 0.3792354, -0.0605250, 0.0605644

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 388
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: A, layer: 1, pos: 400
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: A, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 352
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3081

## Relational analysis of NS_A1_A2_B1_B1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061197, upper bound: 0.0062945
time: 19.78 seconds

## Relational analysis of NS_A1_A2_B1_B1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B1_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0061356, upper bound: 0.0062928
time: 13.21 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1870189, 0.4341814, -0.1871482, 0.4356873, -0.3193238, 0.3196196
1: -1.4449587, -0.2970324, -1.4457805, -0.2968172, -0.5853142, 0.5844216
2: -3.2339847, -2.2606812, -3.2341206, -2.2607253, -0.1051905, 0.1193446
3: -4.1975241, -2.7109001, -4.1991754, -2.7122495, -0.4906029, 0.5097042
4: -2.8547325, -1.4610902, -2.8543129, -1.4621546, -0.2201173, 0.2399446
5: -5.2439280, -3.6427073, -5.2459445, -3.6438558, -0.4473246, 0.4697374
6: -5.8240166, -4.1419382, -5.8247676, -4.1419716, -0.3172389, 0.3244731
7: -2.8017449, -1.2611830, -2.8001790, -1.2619090, -0.4254217, 0.4409451
8: 0.9796946, 1.1543810, 0.9788423, 1.1548324, -0.0173130, 0.0154798
9: -0.0996998, 0.3794597, -0.1008280, 0.3795247, -0.0610596, 0.0619374

Time for backsubstitution: 6.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 373
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 398
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: B, layer: 1, pos: 2830
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 352
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2361

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 3081

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061673, upper bound: 0.0062959
time: 11.01 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0061846, upper bound: 0.0062917
time: 4.32 seconds

## BFS NS instance: NS_A1_A2_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1874416, 0.4341819, -0.1876467, 0.4360710, -0.3200613, 0.3193006
1: -1.4449949, -0.2966350, -1.4476463, -0.2963771, -0.5849919, 0.5866966
2: -3.2358568, -2.2606807, -3.2361946, -2.2591329, -0.1087019, 0.1178689
3: -4.2006512, -2.7108972, -4.2026501, -2.7098446, -0.4960155, 0.5078359
4: -2.8585272, -1.4610896, -2.8585165, -1.4588590, -0.2271757, 0.2369039
5: -5.2471418, -3.6427071, -5.2495189, -3.6414714, -0.4531117, 0.4679928
6: -5.8247471, -4.1419392, -5.8255658, -4.1413574, -0.3186045, 0.3238858
7: -2.8049169, -1.2611824, -2.8037086, -1.2591538, -0.4312130, 0.4386052
8: 0.9796946, 1.1546795, 0.9785895, 1.1551616, -0.0170676, 0.0160455
9: -0.0997031, 0.3796079, -0.1009681, 0.3796895, -0.0609492, 0.0622239

Time for backsubstitution: 6.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 398
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: B, layer: 1, pos: 2830
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 352
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 2842
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2361

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 3081

## Relational analysis of NS_A1_A2_B1_B2_A2_B2_B1

### Relational analysis result of NS_A1_A2_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061864, upper bound: 0.0062945
time: 3.51 seconds

## Relational analysis of NS_A1_A2_B1_B2_A2_B2_B2

### Relational analysis result of NS_A1_A2_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062046, upper bound: 0.0062965
time: 11.99 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1871786, 0.4343462, -0.1874383, 0.4358685, -0.3196202, 0.3200275
1: -1.4449202, -0.2970319, -1.4458308, -0.2966633, -0.5854758, 0.5845988
2: -3.2360005, -2.2606807, -3.2363958, -2.2572949, -0.1104223, 0.1195348
3: -4.1986494, -2.7108846, -4.2009192, -2.7096124, -0.4942300, 0.5109655
4: -2.8572397, -1.4610898, -2.8571289, -1.4579226, -0.2265250, 0.2401565
5: -5.2449493, -3.6427064, -5.2476549, -3.6416087, -0.4503832, 0.4707488
6: -5.8274727, -4.1419392, -5.8286161, -4.1361318, -0.3257756, 0.3247205
7: -2.8013940, -1.2611833, -2.8000309, -1.2612585, -0.4295012, 0.4413675
8: 0.9796947, 1.1546574, 0.9783380, 1.1551408, -0.0173378, 0.0163499
9: -0.0997002, 0.3793380, -0.1007892, 0.3794017, -0.0610643, 0.0623702

Time for backsubstitution: 6.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 398
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: B, layer: 1, pos: 2830
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 352
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 2372
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2361

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 3081

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_B1

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0061869, upper bound: 0.0062911
time: 27.91 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_B1_B2

### Relational analysis result of NS_A1_A2_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062038, upper bound: 0.0062935
time: 7.04 seconds

## BFS NS instance: NS_A2_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1851657, 0.4340776, -0.1855987, 0.4345940, -0.3181606, 0.3167740
1: -1.4421337, -0.2973959, -1.4450312, -0.2970095, -0.5820177, 0.5838650
2: -3.2323108, -2.2595761, -3.2330196, -2.2533882, -0.1204480, 0.1044665
3: -4.1925015, -2.7120285, -4.1938844, -2.7057633, -0.5011511, 0.4901711
4: -2.8500643, -1.4646127, -2.8513422, -1.4560609, -0.2388221, 0.2178552
5: -5.2379341, -3.6438842, -5.2394834, -3.6377482, -0.4583944, 0.4466664
6: -5.8254757, -4.1413808, -5.8256559, -4.1346145, -0.3282787, 0.3155300
7: -2.7959709, -1.2588841, -2.7970891, -1.2539271, -0.4407166, 0.4249988
8: 0.9782586, 1.1539420, 0.9772432, 1.1541150, -0.0142361, 0.0189930
9: -0.0995507, 0.3793142, -0.0996740, 0.3793800, -0.0609092, 0.0608553

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3081
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 3080
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 388
type: B, layer: 1, pos: 388
type: B, layer: 1, pos: 2576
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2166
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 3475
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 373
type: A, layer: 1, pos: 2107
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 403
type: B, layer: 1, pos: 2374
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 3082
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 2809
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 3033
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 400
type: B, layer: 1, pos: 3276
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 2366
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 335
type: B, layer: 1, pos: 3048
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 2799
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 2824
type: B, layer: 1, pos: 2824
type: A, layer: 1, pos: 2373
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 3489
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 3076
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 2117
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2386
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: B, layer: 1, pos: 2362
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3274
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 352
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 2575
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2842

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 3081

## Relational analysis of NS_A2_A1_B2_A1_B2_A1_A1

### Relational analysis result of NS_A2_A1_B2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062931, upper bound: 0.0061418
time: 20.86 seconds

## Relational analysis of NS_A2_A1_B2_A1_B2_A1_A2

### Relational analysis result of NS_A2_A1_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062943, upper bound: 0.0061584
time: 13.68 seconds

## BFS NS instance: NS_A2_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.1856643, 0.4344615, -0.1860215, 0.4345944, -0.3178475, 0.3175413
1: -1.4439995, -0.2969557, -1.4450686, -0.2966123, -0.5842952, 0.5835429
2: -3.2343876, -2.2579830, -3.2348926, -2.2533879, -0.1189684, 0.1079724
3: -4.1959782, -2.7096248, -4.1970110, -2.7057595, -0.4992676, 0.4956593
4: -2.8542686, -1.4613168, -2.8551373, -1.4560599, -0.2357816, 0.2249122
5: -5.2415104, -3.6415024, -5.2426953, -3.6377478, -0.4566413, 0.4524417
6: -5.8262744, -4.1407671, -5.8263860, -4.1346149, -0.3276909, 0.3168941
7: -2.7995000, -1.2561296, -2.8002596, -1.2539265, -0.4383744, 0.4307913
8: 0.9780059, 1.1542714, 0.9772431, 1.1544133, -0.0148068, 0.0187492
9: -0.0996904, 0.3794791, -0.0996774, 0.3795283, -0.0611959, 0.0607449

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3081
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3475
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 373
type: A, layer: 1, pos: 2107
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: A, layer: 1, pos: 2197
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 400
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: A, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2373
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 3489
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 3076
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2830
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 2386
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 352
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 2372
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: A, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 3042
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 2361
type: A, layer: 1, pos: 2361
type: B, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2842

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3081

## Relational analysis of NS_A2_A1_B2_A1_B2_A2_A1

### Relational analysis result of NS_A2_A1_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062938, upper bound: 0.0061528
time: 3.26 seconds

## Relational analysis of NS_A2_A1_B2_A1_B2_A2_A2

### Relational analysis result of NS_A2_A1_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062936, upper bound: 0.0061699
time: 6.21 seconds

## BFS NS instance: NS_A2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1870811, 0.4363371, -0.1870400, 0.4345944, -0.3191057, 0.3205525
1: -1.4465759, -0.2960313, -1.4450706, -0.2959371, -0.5876270, 0.5841979
2: -3.2343168, -2.2575827, -3.2349007, -2.2541149, -0.1219155, 0.1087293
3: -4.1996703, -2.7107630, -4.1970677, -2.7078805, -0.5091705, 0.4966975
4: -2.8550317, -1.4591812, -2.8557541, -1.4560595, -0.2363418, 0.2277753
5: -5.2458572, -3.6426227, -5.2427683, -3.6400080, -0.4683723, 0.4534431
6: -5.8284774, -4.1414022, -5.8263860, -4.1357746, -0.3335585, 0.3184594
7: -2.7982373, -1.2574421, -2.8001866, -1.2570724, -0.4411949, 0.4313082
8: 0.9771814, 1.1545975, 0.9772441, 1.1546438, -0.0158888, 0.0189958
9: -0.1009343, 0.3799258, -0.0996775, 0.3798755, -0.0628557, 0.0611656

Time for backsubstitution: 6.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3081
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 3080
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 388
type: B, layer: 1, pos: 388
type: B, layer: 1, pos: 2576
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 3077
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 2166
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 3475
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 373
type: A, layer: 1, pos: 373
type: A, layer: 1, pos: 2107
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2375
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 403
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 2374
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 3082
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 2809
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 3033
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 400
type: B, layer: 1, pos: 3276
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2366
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 335
type: B, layer: 1, pos: 3048
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 2799
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2373
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 3489
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 3076
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 757
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 3063
type: A, layer: 1, pos: 2117
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 113
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3075
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 147
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 2830
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2386
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 2362
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3274
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 3021
type: B, layer: 1, pos: 558
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 352
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 786
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 2575
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 418
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: A, layer: 1, pos: 2842
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 2361

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3081

## Relational analysis of NS_A2_A1_B2_A2_B2_A2_A1

### Relational analysis result of NS_A2_A1_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062950, upper bound: 0.0062061
time: 11.14 seconds

## Relational analysis of NS_A2_A1_B2_A2_B2_A2_A2

### Relational analysis result of NS_A2_A1_B2_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062921, upper bound: 0.0062272
time: 3.47 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1859978, 0.4344285, -0.1857679, 0.4340296, -0.3165501, 0.3184290
1: -1.4450380, -0.2964975, -1.4432065, -0.2966361, -0.5858561, 0.5810612
2: -3.2346087, -2.2568188, -3.2341986, -2.2584171, -0.1077197, 0.1160426
3: -4.1961102, -2.7084446, -4.1956382, -2.7108121, -0.4909686, 0.5001764
4: -2.8542275, -1.4602917, -2.8535509, -1.4635863, -0.2180463, 0.2393753
5: -5.2422552, -3.6400514, -5.2417774, -3.6423881, -0.4476703, 0.4586709
6: -5.8230391, -4.1404934, -5.8225842, -4.1410818, -0.3173124, 0.3186968
7: -2.8018403, -1.2545774, -2.8014426, -1.2573429, -0.4284585, 0.4380099
8: 0.9777477, 1.1545935, 0.9780000, 1.1545070, -0.0166758, 0.0165272
9: -0.0997177, 0.3796222, -0.0995809, 0.3795554, -0.0613408, 0.0601486

Time for backsubstitution: 6.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3081
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 3080
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: A, layer: 1, pos: 3078
type: B, layer: 1, pos: 388
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 2576
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3077
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2166
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 373
type: A, layer: 1, pos: 373
type: B, layer: 1, pos: 2107
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2375
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 403
type: B, layer: 1, pos: 403
type: A, layer: 1, pos: 2374
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 3082
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2809
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 400
type: A, layer: 1, pos: 3276
type: B, layer: 1, pos: 3276
type: A, layer: 1, pos: 400
type: B, layer: 1, pos: 2366
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 3048
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 2799
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 402
type: B, layer: 1, pos: 402
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2353
type: A, layer: 1, pos: 2353
type: B, layer: 1, pos: 2398
type: A, layer: 1, pos: 2398
type: B, layer: 1, pos: 123
type: A, layer: 1, pos: 123
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2638
type: B, layer: 1, pos: 2638
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3489
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 743
type: A, layer: 1, pos: 743
type: B, layer: 1, pos: 744
type: A, layer: 1, pos: 744
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 3076
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 757
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 3063
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 2117
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 113
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3075
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 147
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 2386
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: A, layer: 1, pos: 774
type: B, layer: 1, pos: 3046
type: A, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2362
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3021
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 786
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3251
type: A, layer: 1, pos: 3251
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 820
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3262
type: A, layer: 1, pos: 3262
type: B, layer: 1, pos: 3492
type: A, layer: 1, pos: 3492
type: B, layer: 1, pos: 418
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 3252
type: B, layer: 1, pos: 3252
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2095
type: B, layer: 1, pos: 2842
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2361

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 3081

## Relational analysis of NS_A2_A2_B1_B1_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061694, upper bound: 0.0062987
time: 3.33 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0061884, upper bound: 0.0062927
time: 29.36 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 34.75 + 1794.64 = 1829.39 seconds
