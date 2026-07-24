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
execution time: IAR + RelationalAnalysis = 7.78 + 26.50 = 34.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0062995, upper bound: 0.0062978

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3475
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 373
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3475

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062289, upper bound: 0.0063006
time: 16.11 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062981, upper bound: 0.0063022
time: 56.74 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 72.91 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 72.91
Output dim: 8, lower bound: -0.0062289, upper bound: 0.0063006
NS_A2, status: Status.UNKNOWN, split count: 1, time: 72.91
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

Time for backsubstitution: 5.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 388
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 400
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3251
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2198

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061686, upper bound: 0.0062996
time: 18.41 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062267, upper bound: 0.0063003
time: 3.66 seconds

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

Time for backsubstitution: 5.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2198
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 388
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 400
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3251
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2198

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062376, upper bound: 0.0062997
time: 58.70 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062979, upper bound: 0.0063005
time: 3.46 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 68.14 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 68.14
Output dim: 8, lower bound: -0.0061686, upper bound: 0.0062996
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 68.14
Output dim: 8, lower bound: -0.0062267, upper bound: 0.0063003
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 68.14
Output dim: 8, lower bound: -0.0062376, upper bound: 0.0062997
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 68.14
Output dim: 8, lower bound: -0.0062979, upper bound: 0.0063005

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.1870837, 0.4343497, -0.1869536, 0.4342483, -0.3196786, 0.3198940
1: -1.4449747, -0.2972406, -1.4440717, -0.2972255, -0.5852007, 0.5843745
2: -3.2344122, -2.2599268, -3.2345545, -2.2606511, -0.1159286, 0.1171485
3: -4.1984086, -2.7086790, -4.1965199, -2.7097697, -0.5015620, 0.5018703
4: -2.8560123, -1.4610900, -2.8553314, -1.4620161, -0.2371224, 0.2390076
5: -5.2442765, -3.6403949, -5.2420769, -3.6417265, -0.4586914, 0.4588655
6: -5.8274908, -4.1405830, -5.8263969, -4.1407561, -0.3234227, 0.3236985
7: -2.8013434, -1.2579926, -2.8007226, -1.2593228, -0.4377789, 0.4386846
8: 0.9796934, 1.1542503, 0.9796720, 1.1545877, -0.0170956, 0.0166899
9: -0.0996990, 0.3795788, -0.0996910, 0.3795928, -0.0612605, 0.0612917

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 373
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 335

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0061679, upper bound: 0.0062796
time: 3.45 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061682, upper bound: 0.0062983
time: 3.56 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.1877643, 0.4343503, -0.1877241, 0.4343807, -0.3206910, 0.3196065
1: -1.4450271, -0.2965720, -1.4451085, -0.2964680, -0.5850763, 0.5858646
2: -3.2379980, -2.2599266, -3.2386150, -2.2594929, -0.1216045, 0.1156872
3: -4.2022772, -2.7086749, -4.2009025, -2.7085855, -0.5069432, 0.5009849
4: -2.8614967, -1.4610891, -2.8615432, -1.4609909, -0.2469747, 0.2364335
5: -5.2486691, -3.6403937, -5.2470546, -3.6402864, -0.4650367, 0.4578460
6: -5.8283319, -4.1405821, -5.8273487, -4.1405044, -0.3246995, 0.3234047
7: -2.8058913, -1.2579912, -2.8058937, -1.2577827, -0.4442211, 0.4374916
8: 0.9796933, 1.1550132, 0.9794137, 1.1554527, -0.0167893, 0.0178536
9: -0.0997034, 0.3796954, -0.0997212, 0.3797233, -0.0612341, 0.0614789

Time for backsubstitution: 5.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 373
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 335

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062278, upper bound: 0.0062784
time: 9.20 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062263, upper bound: 0.0062909
time: 8.01 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.1870815, 0.4345969, -0.1869892, 0.4344659, -0.3199186, 0.3199640
1: -1.4450560, -0.2960308, -1.4440718, -0.2961221, -0.5863997, 0.5843505
2: -3.2350421, -2.2567911, -3.2345552, -2.2579482, -0.1192059, 0.1166589
3: -4.1970520, -2.7083406, -4.1965203, -2.7094712, -0.5029650, 0.5019754
4: -2.8561254, -1.4602911, -2.8553326, -1.4613154, -0.2379094, 0.2389322
5: -5.2426753, -3.6399987, -5.2420778, -3.6413791, -0.4600284, 0.4592400
6: -5.8265123, -4.1402979, -5.8263969, -4.1405087, -0.3250580, 0.3234058
7: -2.8013644, -1.2545317, -2.8007233, -1.2560693, -0.4412398, 0.4385189
8: 0.9777474, 1.1546926, 0.9780052, 1.1545877, -0.0167057, 0.0188134
9: -0.0997168, 0.3800923, -0.0996909, 0.3800775, -0.0619659, 0.0611627

Time for backsubstitution: 5.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 373
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 335

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062378, upper bound: 0.0062774
time: 25.31 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062385, upper bound: 0.0062958
time: 14.17 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.1877619, 0.4345976, -0.1877601, 0.4345982, -0.3209311, 0.3196763
1: -1.4451090, -0.2953620, -1.4451085, -0.2953646, -0.5862755, 0.5858406
2: -3.2386286, -2.2567911, -3.2386150, -2.2567894, -0.1248816, 0.1151978
3: -4.2009201, -2.7083364, -4.2009029, -2.7082865, -0.5083444, 0.5010923
4: -2.8616092, -1.4602904, -2.8615441, -1.4602904, -0.2477618, 0.2363581
5: -5.2470675, -3.6399975, -5.2470546, -3.6399388, -0.4663714, 0.4582230
6: -5.8273535, -4.1402969, -5.8273487, -4.1402559, -0.3263348, 0.3231119
7: -2.8059120, -1.2545302, -2.8058937, -1.2545291, -0.4476820, 0.4373259
8: 0.9777473, 1.1554558, 0.9777470, 1.1554528, -0.0163995, 0.0199771
9: -0.0997214, 0.3802087, -0.0997212, 0.3802079, -0.0619395, 0.0613500

Time for backsubstitution: 5.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 373
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 335

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062962, upper bound: 0.0062775
time: 39.96 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062963, upper bound: 0.0062949
time: 7.02 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 53.01 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 53.01
Output dim: 8, lower bound: -0.0061679, upper bound: 0.0062796
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 53.01
Output dim: 8, lower bound: -0.0061682, upper bound: 0.0062983
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 53.01
Output dim: 8, lower bound: -0.0062278, upper bound: 0.0062784
NS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 53.01
Output dim: 8, lower bound: -0.0062263, upper bound: 0.0062909
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 53.01
Output dim: 8, lower bound: -0.0062378, upper bound: 0.0062774
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 53.01
Output dim: 8, lower bound: -0.0062385, upper bound: 0.0062958
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 53.01
Output dim: 8, lower bound: -0.0062962, upper bound: 0.0062775
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 53.01
Output dim: 8, lower bound: -0.0062963, upper bound: 0.0062949

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.1871301, 0.4343465, -0.1868928, 0.4342448, -0.3195106, 0.3196473
1: -1.4449914, -0.2970874, -1.4440037, -0.2972257, -0.5853211, 0.5844828
2: -3.2344251, -2.2564967, -3.2345352, -2.2606511, -0.1139551, 0.1203970
3: -4.1985421, -2.7060623, -4.1961527, -2.7097733, -0.5008790, 0.5036145
4: -2.8560131, -1.4568583, -2.8553061, -1.4620161, -0.2346691, 0.2429651
5: -5.2445130, -3.6381516, -5.2417283, -3.6417272, -0.4583420, 0.4606087
6: -5.8274188, -4.1347442, -5.8263245, -4.1407557, -0.3201093, 0.3289342
7: -2.8003597, -1.2573425, -2.7996097, -1.2593229, -0.4364892, 0.4411787
8: 0.9791891, 1.1542379, 0.9796718, 1.1545701, -0.0176398, 0.0163575
9: -0.0996598, 0.3793958, -0.0996909, 0.3794168, -0.0615317, 0.0611177

Time for backsubstitution: 5.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 388
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 400
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3251
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2197

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061482, upper bound: 0.0062979
time: 11.03 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061596, upper bound: 0.0062952
time: 8.45 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.1871281, 0.4345940, -0.1869287, 0.4344624, -0.3197504, 0.3197172
1: -1.4450725, -0.2958776, -1.4440038, -0.2961224, -0.5865198, 0.5844590
2: -3.2350559, -2.2533612, -3.2345347, -2.2579482, -0.1172320, 0.1199074
3: -4.1971850, -2.7057242, -4.1961536, -2.7094750, -0.5022775, 0.5037197
4: -2.8561254, -1.4560597, -2.8553076, -1.4613155, -0.2354563, 0.2428897
5: -5.2429123, -3.6377549, -5.2417288, -3.6413801, -0.4596734, 0.4609834
6: -5.8264389, -4.1344590, -5.8263254, -4.1405087, -0.3217446, 0.3286414
7: -2.8003812, -1.2538812, -2.7996104, -1.2560693, -0.4399504, 0.4410131
8: 0.9772432, 1.1546804, 0.9780052, 1.1545703, -0.0172500, 0.0184810
9: -0.0996778, 0.3799092, -0.0996909, 0.3799016, -0.0622371, 0.0609888

Time for backsubstitution: 5.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 388
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 400
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3251
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2197

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062182, upper bound: 0.0062977
time: 12.14 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062291, upper bound: 0.0062952
time: 43.31 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1875188, 0.4344133, -0.1875396, 0.4344300, -0.3203560, 0.3191333
1: -1.4450755, -0.2953627, -1.4450785, -0.2953653, -0.5862187, 0.5857872
2: -3.2363653, -2.2567909, -3.2365794, -2.2567897, -0.1227172, 0.1132146
3: -4.1992860, -2.7083580, -4.1994109, -2.7083061, -0.5064043, 0.4992077
4: -2.8587933, -1.4602909, -2.8590119, -1.4602906, -0.2450971, 0.2339079
5: -5.2455673, -3.6399996, -5.2456846, -3.6399407, -0.4650062, 0.4569079
6: -5.8234320, -4.1402979, -5.8238220, -4.1402569, -0.3227743, 0.3198107
7: -2.8050814, -1.2545304, -2.8051317, -1.2545294, -0.4459704, 0.4357406
8: 0.9777474, 1.1551349, 0.9777470, 1.1551589, -0.0160737, 0.0196198
9: -0.0997212, 0.3801482, -0.0997211, 0.3801535, -0.0617778, 0.0611713

Time for backsubstitution: 5.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 388
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 400
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3251
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2197

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062377, upper bound: 0.0062757
time: 7.85 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062954, upper bound: 0.0062732
time: 20.78 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1878087, 0.4345947, -0.1876996, 0.4345948, -0.3207633, 0.3194293
1: -1.4451256, -0.2952088, -1.4450401, -0.2953649, -0.5863959, 0.5859490
2: -3.2386422, -2.2533610, -3.2385948, -2.2567897, -0.1229077, 0.1184463
3: -4.2010536, -2.7057199, -4.2005363, -2.7082903, -0.5076579, 0.5028354
4: -2.8616090, -1.4560583, -2.8615186, -1.4602903, -0.2453088, 0.2403156
5: -5.2473044, -3.6377537, -5.2467060, -3.6399403, -0.4660178, 0.4599658
6: -5.8272805, -4.1344595, -5.8272772, -4.1402564, -0.3230214, 0.3283476
7: -2.8049288, -1.2538797, -2.8047814, -1.2545291, -0.4463931, 0.4398201
8: 0.9772432, 1.1554434, 0.9777470, 1.1554352, -0.0169437, 0.0196446
9: -0.0996822, 0.3800254, -0.0997212, 0.3800317, -0.0622106, 0.0611760

Time for backsubstitution: 5.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2197
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 388
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 400
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3251
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2197

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062361, upper bound: 0.0062949
time: 3.11 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062940, upper bound: 0.0062909
time: 10.72 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 19.87 seconds
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.87
Output dim: 8, lower bound: -0.0061482, upper bound: 0.0062979
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.87
Output dim: 8, lower bound: -0.0061596, upper bound: 0.0062952
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.87
Output dim: 8, lower bound: -0.0062182, upper bound: 0.0062977
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.87
Output dim: 8, lower bound: -0.0062291, upper bound: 0.0062952
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 19.87
Output dim: 8, lower bound: -0.0062377, upper bound: 0.0062757
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.87
Output dim: 8, lower bound: -0.0062954, upper bound: 0.0062732
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.87
Output dim: 8, lower bound: -0.0062361, upper bound: 0.0062949
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.87
Output dim: 8, lower bound: -0.0062940, upper bound: 0.0062909

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1865214, 0.4343464, -0.1862086, 0.4341657, -0.3186493, 0.3189571
1: -1.4449534, -0.2976520, -1.4433728, -0.2978700, -0.5846478, 0.5833977
2: -3.2317500, -2.2564969, -3.2314854, -2.2611742, -0.1098526, 0.1168574
3: -4.1958079, -2.7060666, -4.1930413, -2.7105086, -0.4972201, 0.5004554
4: -2.8506417, -1.4568590, -2.8491979, -1.4630690, -0.2263081, 0.2359124
5: -5.2413998, -3.6381528, -5.2381759, -3.6425633, -0.4540661, 0.4569047
6: -5.8265748, -4.1347446, -5.8253665, -4.1409202, -0.3189820, 0.3278649
7: -2.7969246, -1.2573428, -2.7956798, -1.2602813, -0.4319122, 0.4371486
8: 0.9791892, 1.1535454, 0.9798648, 1.1537790, -0.0168011, 0.0153549
9: -0.0996564, 0.3792801, -0.0997097, 0.3792892, -0.0613723, 0.0609391

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 373
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 398

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0061462, upper bound: 0.0062040
time: 42.66 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0061464, upper bound: 0.0062930
time: 6.97 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1870882, 0.4343465, -0.1868441, 0.4342447, -0.3194777, 0.3187142
1: -1.4449898, -0.2971010, -1.4440016, -0.2972416, -0.5845392, 0.5844809
2: -3.2343071, -2.2564967, -3.2343988, -2.2606514, -0.1139494, 0.1158883
3: -4.1984797, -2.7060640, -4.1960812, -2.7097754, -0.5008109, 0.4996781
4: -2.8557286, -1.4568584, -2.8549790, -1.4620161, -0.2346523, 0.2338678
5: -5.2444420, -3.6381521, -5.2416468, -3.6417274, -0.4583174, 0.4562199
6: -5.8273821, -4.1347446, -5.8262835, -4.1407566, -0.3201079, 0.3275755
7: -2.8002768, -1.2573421, -2.7995141, -1.2593231, -0.4364474, 0.4363810
8: 0.9791891, 1.1542211, 0.9796718, 1.1545507, -0.0165523, 0.0163555
9: -0.0996596, 0.3793877, -0.0996907, 0.3794076, -0.0613432, 0.0611163

Time for backsubstitution: 6.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 373
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 398

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0061576, upper bound: 0.0062288
time: 21.85 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061584, upper bound: 0.0062935
time: 8.15 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1865191, 0.4345936, -0.1862444, 0.4343834, -0.3188893, 0.3190269
1: -1.4450338, -0.2964419, -1.4433732, -0.2967666, -0.5858466, 0.5833740
2: -3.2323816, -2.2533612, -3.2314856, -2.2584713, -0.1131297, 0.1163677
3: -4.1944504, -2.7057281, -4.1930413, -2.7102098, -0.4986195, 0.5005593
4: -2.8507540, -1.4560603, -2.8491983, -1.4623685, -0.2270954, 0.2358370
5: -5.2397985, -3.6377563, -5.2381763, -3.6422167, -0.4553985, 0.4572778
6: -5.8255968, -4.1344595, -5.8253665, -4.1406717, -0.3206173, 0.3275721
7: -2.7969456, -1.2538817, -2.7956805, -1.2570276, -0.4353731, 0.4369831
8: 0.9772433, 1.1539880, 0.9781981, 1.1537791, -0.0164112, 0.0174784
9: -0.0996744, 0.3797934, -0.0997097, 0.3797736, -0.0620777, 0.0608101

Time for backsubstitution: 6.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 373
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 398

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062159, upper bound: 0.0062245
time: 14.48 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062145, upper bound: 0.0062931
time: 12.30 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1870859, 0.4345940, -0.1868800, 0.4344624, -0.3197177, 0.3187842
1: -1.4450709, -0.2958913, -1.4440016, -0.2961380, -0.5857379, 0.5844573
2: -3.2349381, -2.2533612, -3.2343986, -2.2579484, -0.1172263, 0.1153987
3: -4.1971235, -2.7057257, -4.1960816, -2.7094765, -0.5022094, 0.4997832
4: -2.8558416, -1.4560595, -2.8549800, -1.4613156, -0.2354395, 0.2337924
5: -5.2428417, -3.6377554, -5.2416482, -3.6413803, -0.4596490, 0.4565944
6: -5.8264036, -4.1344595, -5.8262835, -4.1405087, -0.3217433, 0.3272826
7: -2.8002980, -1.2538810, -2.7995145, -1.2560693, -0.4399083, 0.4362154
8: 0.9772432, 1.1546638, 0.9780051, 1.1545509, -0.0161625, 0.0184790
9: -0.0996775, 0.3799011, -0.0996908, 0.3798922, -0.0620486, 0.0609873

Time for backsubstitution: 6.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 373
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 398

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062295, upper bound: 0.0062282
time: 34.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062286, upper bound: 0.0062954
time: 71.43 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1874766, 0.4344133, -0.1874910, 0.4344300, -0.3203324, 0.3182116
1: -1.4450743, -0.2953764, -1.4450771, -0.2953811, -0.5854364, 0.5857858
2: -3.2363033, -2.2567911, -3.2365081, -2.2567897, -0.1227118, 0.1087826
3: -4.1992240, -2.7083592, -4.1993399, -2.7083077, -0.5063472, 0.4955560
4: -2.8585091, -1.4602903, -2.8586838, -1.4602906, -0.2450808, 0.2248925
5: -5.2454977, -3.6400001, -5.2456026, -3.6399412, -0.4649820, 0.4525851
6: -5.8233962, -4.1402974, -5.8237810, -4.1402569, -0.3227727, 0.3185828
7: -2.8049989, -1.2545303, -2.8050361, -1.2545294, -0.4459383, 0.4310183
8: 0.9777474, 1.1551182, 0.9777470, 1.1551396, -0.0149966, 0.0196178
9: -0.0997210, 0.3801401, -0.0997211, 0.3801443, -0.0615958, 0.0611703

Time for backsubstitution: 6.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 373
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 398

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062945, upper bound: 0.0062111
time: 3.62 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062930, upper bound: 0.0062044
time: 50.14 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1871994, 0.4345942, -0.1870140, 0.4345157, -0.3198736, 0.3187381
1: -1.4450862, -0.2957732, -1.4443585, -0.2960093, -0.5857226, 0.5847586
2: -3.2358794, -2.2533612, -3.2354484, -2.2574716, -0.1187300, 0.1149061
3: -4.1983175, -2.7057238, -4.1974120, -2.7090392, -0.5037715, 0.4996949
4: -2.8562384, -1.4560587, -2.8554087, -1.4613426, -0.2368456, 0.2332612
5: -5.2441893, -3.6377554, -5.2431488, -3.6407790, -0.4616573, 0.4562491
6: -5.8264389, -4.1344585, -5.8263197, -4.1404200, -0.3217530, 0.3272772
7: -2.8014936, -1.2538800, -2.8008497, -1.2554876, -0.4416143, 0.4357852
8: 0.9772433, 1.1547508, 0.9779400, 1.1546440, -0.0161049, 0.0186323
9: -0.0996790, 0.3799049, -0.0997013, 0.3798938, -0.0620506, 0.0609873

Time for backsubstitution: 6.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 373
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 398

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062346, upper bound: 0.0062264
time: 19.57 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062356, upper bound: 0.0062977
time: 3.34 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1877665, 0.4345946, -0.1876507, 0.4345948, -0.3207398, 0.3185079
1: -1.4451243, -0.2952225, -1.4450386, -0.2953806, -0.5856137, 0.5859478
2: -3.2385802, -2.2533607, -3.2385230, -2.2567892, -0.1229023, 0.1140143
3: -4.2009921, -2.7057219, -4.2004652, -2.7082915, -0.5076011, 0.4991832
4: -2.8613250, -1.4560585, -2.8611913, -1.4602902, -0.2452926, 0.2313002
5: -5.2472343, -3.6377540, -5.2466249, -3.6399403, -0.4659935, 0.4556429
6: -5.8272457, -4.1344585, -5.8272362, -4.1402559, -0.3230200, 0.3271196
7: -2.8048456, -1.2538795, -2.8046856, -1.2545292, -0.4463609, 0.4350976
8: 0.9772432, 1.1554266, 0.9777470, 1.1554159, -0.0158667, 0.0196426
9: -0.0996820, 0.3800173, -0.0997212, 0.3800225, -0.0620287, 0.0611750

Time for backsubstitution: 6.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 398
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 373
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 398

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062935, upper bound: 0.0062265
time: 4.16 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062936, upper bound: 0.0062988
time: 3.31 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 13.88 seconds
NS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 13.88
Output dim: 8, lower bound: -0.0061462, upper bound: 0.0062040
NS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 13.88
Output dim: 8, lower bound: -0.0061464, upper bound: 0.0062930
NS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 13.88
Output dim: 8, lower bound: -0.0061576, upper bound: 0.0062288
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -0.0061584, upper bound: 0.0062935
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 13.88
Output dim: 8, lower bound: -0.0062159, upper bound: 0.0062245
NS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 13.88
Output dim: 8, lower bound: -0.0062145, upper bound: 0.0062931
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 13.88
Output dim: 8, lower bound: -0.0062295, upper bound: 0.0062282
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -0.0062286, upper bound: 0.0062954
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -0.0062945, upper bound: 0.0062111
NS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 13.88
Output dim: 8, lower bound: -0.0062930, upper bound: 0.0062044
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 13.88
Output dim: 8, lower bound: -0.0062346, upper bound: 0.0062264
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -0.0062356, upper bound: 0.0062977
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -0.0062935, upper bound: 0.0062265
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.88
Output dim: 8, lower bound: -0.0062936, upper bound: 0.0062988

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1872912, 0.4362218, -0.1868011, 0.4342447, -0.3194862, 0.3206291
1: -1.4475621, -0.2969908, -1.4440016, -0.2972795, -0.5871356, 0.5842954
2: -3.2342234, -2.2561378, -3.2343976, -2.2614074, -0.1159006, 0.1155778
3: -4.2020555, -2.7073002, -4.1960745, -2.7119718, -0.5099699, 0.4996930
4: -2.8557475, -1.4547263, -2.8549457, -1.4620162, -0.2344975, 0.2360628
5: -5.2486463, -3.6393352, -5.2416382, -3.6440382, -0.4693577, 0.4561314
6: -5.8295698, -4.1355963, -5.8262825, -4.1421204, -0.3247470, 0.3272343
7: -2.7990026, -1.2587150, -2.7994306, -1.2625158, -0.4386957, 0.4361954
8: 0.9783649, 1.1542697, 0.9796730, 1.1545378, -0.0173733, 0.0163397
9: -0.1009065, 0.3794207, -0.0996907, 0.3793834, -0.0626201, 0.0610980

Time for backsubstitution: 6.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 388
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 400
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3251
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 158

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061487, upper bound: 0.0062971
time: 3.92 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061556, upper bound: 0.0062957
time: 4.50 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1872891, 0.4364692, -0.1868369, 0.4344624, -0.3197263, 0.3206990
1: -1.4476430, -0.2957805, -1.4440017, -0.2961762, -0.5883343, 0.5842713
2: -3.2348547, -2.2530024, -3.2343974, -2.2587044, -0.1191742, 0.1150882
3: -4.2007008, -2.7069621, -4.1960745, -2.7116728, -0.5113304, 0.4997984
4: -2.8558602, -1.4539273, -2.8549466, -1.4613155, -0.2352847, 0.2359874
5: -5.2470479, -3.6389382, -5.2416382, -3.6436915, -0.4706461, 0.4565061
6: -5.8285913, -4.1353111, -5.8262825, -4.1418729, -0.3263819, 0.3269415
7: -2.7990234, -1.2552537, -2.7994311, -1.2592621, -0.4421567, 0.4360300
8: 0.9764191, 1.1547123, 0.9780064, 1.1545379, -0.0169834, 0.0184632
9: -0.1009245, 0.3799343, -0.0996908, 0.3798680, -0.0633255, 0.0609691

Time for backsubstitution: 6.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 388
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 400
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3251
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 158

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062210, upper bound: 0.0062746
time: 60.00 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062245, upper bound: 0.0062969
time: 4.21 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.1862631, 0.4344128, -0.1864292, 0.4344296, -0.3190809, 0.3171154
1: -1.4450717, -0.2961897, -1.4450750, -0.2960942, -0.5846980, 0.5849512
2: -3.2362928, -2.2568271, -3.2364984, -2.2568181, -0.1217797, 0.1076011
3: -4.1991525, -2.7084577, -4.1992712, -2.7083912, -0.5055600, 0.4945975
4: -2.8577647, -1.4602914, -2.8580325, -1.4602911, -0.2444330, 0.2242512
5: -5.2454047, -3.6400633, -5.2455158, -3.6399927, -0.4642536, 0.4515935
6: -5.8233953, -4.1405087, -5.8237820, -4.1404524, -0.3213787, 0.3168297
7: -2.8049817, -1.2545894, -2.8050213, -1.2545758, -0.4453297, 0.4304504
8: 0.9777477, 1.1548409, 0.9777473, 1.1548960, -0.0147549, 0.0193590
9: -0.0997207, 0.3797202, -0.0997208, 0.3797713, -0.0612118, 0.0607322

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 388
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 400
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3251
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 158

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062759, upper bound: 0.0062090
time: 3.42 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062942, upper bound: 0.0062062
time: 3.74 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.1874023, 0.4364692, -0.1869707, 0.4345157, -0.3198813, 0.3206502
1: -1.4476579, -0.2956637, -1.4443586, -0.2960476, -0.5883186, 0.5845821
2: -3.2357910, -2.2530017, -3.2354462, -2.2582266, -0.1206615, 0.1146012
3: -4.2018042, -2.7069604, -4.1974020, -2.7112389, -0.5126969, 0.4997355
4: -2.8562577, -1.4539268, -2.8553751, -1.4613426, -0.2366890, 0.2354679
5: -5.2482896, -3.6389370, -5.2431355, -3.6430895, -0.4724771, 0.4561800
6: -5.8286343, -4.1353111, -5.8263178, -4.1417809, -0.3260979, 0.3269718
7: -2.8002188, -1.2552506, -2.8007617, -1.2586789, -0.4438980, 0.4355749
8: 0.9764191, 1.1547992, 0.9779413, 1.1546308, -0.0169307, 0.0186194
9: -0.1009259, 0.3799393, -0.0997013, 0.3798718, -0.0633252, 0.0609722

Time for backsubstitution: 6.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 388
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 400
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3251
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 158

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062196, upper bound: 0.0062943
time: 3.40 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062367, upper bound: 0.0062962
time: 35.36 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.1865529, 0.4345944, -0.1865890, 0.4345945, -0.3194885, 0.3174116
1: -1.4451219, -0.2960357, -1.4450366, -0.2960939, -0.5848753, 0.5851129
2: -3.2385700, -2.2533967, -3.2385135, -2.2568181, -0.1219701, 0.1128328
3: -4.2009187, -2.7058196, -4.2003956, -2.7083757, -0.5068144, 0.4982226
4: -2.8605807, -1.4560590, -2.8605394, -1.4602909, -0.2446447, 0.2306590
5: -5.2471399, -3.6378176, -5.2465363, -3.6399918, -0.4652647, 0.4546494
6: -5.8272457, -4.1346693, -5.8272357, -4.1404524, -0.3216261, 0.3253663
7: -2.8048286, -1.2539389, -2.8046703, -1.2545757, -0.4457525, 0.4345297
8: 0.9772434, 1.1551495, 0.9777473, 1.1551725, -0.0156249, 0.0193838
9: -0.0996819, 0.3795972, -0.0997210, 0.3796496, -0.0616446, 0.0607369

Time for backsubstitution: 6.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 388
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 400
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3251
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 158

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062761, upper bound: 0.0062282
time: 47.31 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062940, upper bound: 0.0062281
time: 5.30 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1879692, 0.4364695, -0.1876074, 0.4345949, -0.3207487, 0.3204197
1: -1.4476967, -0.2951133, -1.4450386, -0.2954188, -0.5882104, 0.5857703
2: -3.2384906, -2.2530012, -3.2385216, -2.2575436, -0.1247809, 0.1135877
3: -4.2044296, -2.7069578, -4.2004509, -2.7104936, -0.5164626, 0.4992637
4: -2.8613439, -1.4539263, -2.8611567, -1.4602908, -0.2451358, 0.2335186
5: -5.2512789, -3.6389360, -5.2466083, -3.6422501, -0.4767091, 0.4556525
6: -5.8294277, -4.1353102, -5.8272352, -4.1416121, -0.3271095, 0.3269298
7: -2.8035705, -1.2552503, -2.8045952, -1.2577201, -0.4485206, 0.4349530
8: 0.9764190, 1.1554748, 0.9777483, 1.1554027, -0.0166974, 0.0196301
9: -0.1009292, 0.3800516, -0.0997212, 0.3800006, -0.0633043, 0.0611598

Time for backsubstitution: 6.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 388
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 400
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3251
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 158

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062738, upper bound: 0.0062953
time: 81.73 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062926, upper bound: 0.0062962
time: 169.91 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 258.02 seconds
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 258.02
Output dim: 8, lower bound: -0.0061487, upper bound: 0.0062971
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 258.02
Output dim: 8, lower bound: -0.0061556, upper bound: 0.0062957
NS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 258.02
Output dim: 8, lower bound: -0.0062210, upper bound: 0.0062746
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 258.02
Output dim: 8, lower bound: -0.0062245, upper bound: 0.0062969
NS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 258.02
Output dim: 8, lower bound: -0.0062759, upper bound: 0.0062090
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 258.02
Output dim: 8, lower bound: -0.0062942, upper bound: 0.0062062
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 258.02
Output dim: 8, lower bound: -0.0062196, upper bound: 0.0062943
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 258.02
Output dim: 8, lower bound: -0.0062367, upper bound: 0.0062962
NS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 258.02
Output dim: 8, lower bound: -0.0062761, upper bound: 0.0062282
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 258.02
Output dim: 8, lower bound: -0.0062940, upper bound: 0.0062281
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 258.02
Output dim: 8, lower bound: -0.0062738, upper bound: 0.0062953
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 258.02
Output dim: 8, lower bound: -0.0062926, upper bound: 0.0062962

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1868662, 0.4362213, -0.1863003, 0.4338607, -0.3187063, 0.3202123
1: -1.4475253, -0.2973937, -1.4421359, -0.2977259, -0.5866876, 0.5820179
2: -3.2323332, -2.2561383, -3.2323027, -2.2630005, -0.1123979, 0.1134689
3: -4.1989021, -2.7073040, -4.1925602, -2.7143755, -0.5043793, 0.4961482
4: -2.8519430, -1.4547268, -2.8507316, -1.4653132, -0.2274380, 0.2318590
5: -5.2453933, -3.6393366, -5.2380104, -3.6464212, -0.4635617, 0.4524222
6: -5.8288307, -4.1355968, -5.8254709, -4.1427326, -0.3233730, 0.3264103
7: -2.7958210, -1.2587157, -2.7958918, -1.2652706, -0.4328706, 0.4327123
8: 0.9783649, 1.1539670, 0.9799258, 1.1542039, -0.0170394, 0.0157671
9: -0.1009034, 0.3792719, -0.0995507, 0.3792179, -0.0624531, 0.0608119

Time for backsubstitution: 6.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 373
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3081

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0061500, upper bound: 0.0062763
time: 7.12 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0061484, upper bound: 0.0062765
time: 35.82 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1872889, 0.4362217, -0.1867986, 0.4342447, -0.3194800, 0.3199060
1: -1.4475617, -0.2969967, -1.4440013, -0.2972859, -0.5863641, 0.5842945
2: -3.2342062, -2.2561386, -3.2343786, -2.2614074, -0.1158929, 0.1119819
3: -4.2020226, -2.7073011, -4.1960382, -2.7119718, -0.5099162, 0.4942631
4: -2.8557377, -1.4547263, -2.8549347, -1.4620162, -0.2344870, 0.2288101
5: -5.2486000, -3.6393356, -5.2415876, -3.6440384, -0.4693334, 0.4506618
6: -5.8295588, -4.1355972, -5.8262696, -4.1421204, -0.3247367, 0.3258207
7: -2.7989931, -1.2587160, -2.7994201, -1.2625157, -0.4386581, 0.4303719
8: 0.9783649, 1.1542653, 0.9796731, 1.1545330, -0.0167966, 0.0163372
9: -0.1009066, 0.3794201, -0.0996907, 0.3793828, -0.0623426, 0.0610966

Time for backsubstitution: 6.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 373
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3081

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0061532, upper bound: 0.0062769
time: 9.17 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061547, upper bound: 0.0062958
time: 3.83 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1872867, 0.4364692, -0.1868342, 0.4344624, -0.3197199, 0.3199758
1: -1.4476429, -0.2957864, -1.4440018, -0.2961825, -0.5875630, 0.5842703
2: -3.2348368, -2.2530031, -3.2343786, -2.2587049, -0.1191665, 0.1114923
3: -4.2006683, -2.7069633, -4.1960387, -2.7116728, -0.5112765, 0.4943685
4: -2.8558502, -1.4539273, -2.8549361, -1.4613156, -0.2352741, 0.2287347
5: -5.2470021, -3.6389389, -5.2415876, -3.6436906, -0.4706217, 0.4510365
6: -5.8285799, -4.1353121, -5.8262696, -4.1418724, -0.3263716, 0.3255279
7: -2.7990141, -1.2552547, -2.7994208, -1.2592622, -0.4421192, 0.4302064
8: 0.9764191, 1.1547079, 0.9780064, 1.1545331, -0.0164068, 0.0184606
9: -0.1009244, 0.3799338, -0.0996908, 0.3798674, -0.0630480, 0.0609677

Time for backsubstitution: 6.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 373
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3081

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062245, upper bound: 0.0062738
time: 34.00 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062234, upper bound: 0.0062947
time: 37.64 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1862607, 0.4344128, -0.1864265, 0.4344296, -0.3190777, 0.3163845
1: -1.4450721, -0.2961954, -1.4450747, -0.2961006, -0.5839213, 0.5849501
2: -3.2362742, -2.2568269, -3.2364783, -2.2568176, -0.1217733, 0.1040110
3: -4.1991196, -2.7084577, -4.1992345, -2.7083912, -0.5055060, 0.4893931
4: -2.8577547, -1.4602911, -2.8580213, -1.4602910, -0.2444206, 0.2170004
5: -5.2453585, -3.6400633, -5.2454643, -3.6399922, -0.4642303, 0.4461401
6: -5.8233843, -4.1405091, -5.8237686, -4.1404524, -0.3213684, 0.3154180
7: -2.8049722, -1.2545893, -2.8050106, -1.2545760, -0.4452933, 0.4246252
8: 0.9777477, 1.1548367, 0.9777473, 1.1548913, -0.0141829, 0.0193569
9: -0.0997208, 0.3797197, -0.0997208, 0.3797709, -0.0609341, 0.0607315

Time for backsubstitution: 6.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 373
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3081

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062930, upper bound: 0.0061890
time: 10.79 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062929, upper bound: 0.0062115
time: 3.67 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.1869772, 0.4364686, -0.1864699, 0.4341317, -0.3190973, 0.3202285
1: -1.4476207, -0.2960666, -1.4424931, -0.2964939, -0.5878704, 0.5823046
2: -3.2339001, -2.2530022, -3.2333508, -2.2598197, -0.1171589, 0.1124922
3: -4.1986499, -2.7069638, -4.1938872, -2.7136431, -0.5070591, 0.4962052
4: -2.8524520, -1.4539273, -2.8511605, -1.4646386, -0.2296396, 0.2312643
5: -5.2450371, -3.6389384, -5.2395082, -3.6454720, -0.4666837, 0.4524687
6: -5.8278933, -4.1353116, -5.8255081, -4.1423936, -0.3247238, 0.3261478
7: -2.7970376, -1.2552522, -2.7972224, -1.2614331, -0.4380728, 0.4320919
8: 0.9764190, 1.1544966, 0.9781939, 1.1542969, -0.0165997, 0.0180463
9: -0.1009227, 0.3797904, -0.0995615, 0.3797066, -0.0631582, 0.0606860

Time for backsubstitution: 6.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 373
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3081

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062189, upper bound: 0.0062766
time: 6.31 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062192, upper bound: 0.0062932
time: 99.47 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.1873999, 0.4364693, -0.1869682, 0.4345157, -0.3198783, 0.3199227
1: -1.4476579, -0.2956695, -1.4443586, -0.2960539, -0.5875466, 0.5845813
2: -3.2357731, -2.2530022, -3.2354264, -2.2582269, -0.1206538, 0.1110031
3: -4.2017713, -2.7069612, -4.1973648, -2.7112389, -0.5126433, 0.4944209
4: -2.8562479, -1.4539268, -2.8553643, -1.4613426, -0.2366777, 0.2282152
5: -5.2482443, -3.6389375, -5.2430849, -3.6430898, -0.4724530, 0.4507087
6: -5.8286223, -4.1353121, -5.8263044, -4.1417809, -0.3260874, 0.3255581
7: -2.8002095, -1.2552516, -2.8007512, -1.2586787, -0.4438605, 0.4297513
8: 0.9764191, 1.1547949, 0.9779413, 1.1546260, -0.0163602, 0.0186175
9: -0.1009259, 0.3799388, -0.0997012, 0.3798714, -0.0630477, 0.0609707

Time for backsubstitution: 6.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 373
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3081

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062337, upper bound: 0.0062784
time: 3.61 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062323, upper bound: 0.0062949
time: 26.60 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.1865507, 0.4345943, -0.1865864, 0.4345944, -0.3194853, 0.3166806
1: -1.4451218, -0.2960414, -1.4450365, -0.2961003, -0.5840984, 0.5851117
2: -3.2385521, -2.2533970, -3.2384937, -2.2568181, -0.1219637, 0.1092427
3: -4.2008853, -2.7058194, -4.2003593, -2.7083757, -0.5067605, 0.4930170
4: -2.8605709, -1.4560590, -2.8605289, -1.4602907, -0.2446322, 0.2234082
5: -5.2470946, -3.6378174, -5.2464857, -3.6399915, -0.4652416, 0.4491959
6: -5.8272333, -4.1346693, -5.8272238, -4.1404514, -0.3216157, 0.3239546
7: -2.8048193, -1.2539389, -2.8046603, -1.2545758, -0.4457158, 0.4287044
8: 0.9772434, 1.1551454, 0.9777473, 1.1551677, -0.0150530, 0.0193817
9: -0.0996819, 0.3795968, -0.0997210, 0.3796492, -0.0613669, 0.0607362

Time for backsubstitution: 6.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 373
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3081

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062924, upper bound: 0.0062136
time: 3.37 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062938, upper bound: 0.0062285
time: 3.73 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.1875441, 0.4364688, -0.1871065, 0.4342107, -0.3199646, 0.3200005
1: -1.4476595, -0.2955163, -1.4431734, -0.2958652, -0.5877619, 0.5834929
2: -3.2366009, -2.2530017, -3.2364266, -2.2591348, -0.1212784, 0.1114787
3: -4.2012753, -2.7069614, -4.1969380, -2.7128978, -0.5108248, 0.4957459
4: -2.8575397, -1.4539268, -2.8569429, -1.4635859, -0.2380861, 0.2293149
5: -5.2480259, -3.6389368, -5.2429805, -3.6446333, -0.4709157, 0.4519410
6: -5.8286862, -4.1353106, -5.8264236, -4.1422248, -0.3257355, 0.3261057
7: -2.8003893, -1.2552512, -2.8010569, -1.2604746, -0.4426955, 0.4314699
8: 0.9764191, 1.1551723, 0.9780011, 1.1550686, -0.0163677, 0.0190570
9: -0.1009261, 0.3799028, -0.0995812, 0.3798351, -0.0631373, 0.0608737

Time for backsubstitution: 6.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 373
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3081

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062753, upper bound: 0.0062780
time: 33.73 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062749, upper bound: 0.0062920
time: 26.78 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.1879668, 0.4364694, -0.1876050, 0.4345949, -0.3207456, 0.3196946
1: -1.4476963, -0.2951191, -1.4450384, -0.2954252, -0.5874383, 0.5857695
2: -3.2384720, -2.2530022, -3.2385018, -2.2575438, -0.1247733, 0.1099905
3: -4.2043962, -2.7069588, -4.2004156, -2.7104936, -0.5164090, 0.4940549
4: -2.8613338, -1.4539266, -2.8611462, -1.4602908, -0.2451246, 0.2262659
5: -5.2512331, -3.6389372, -5.2465572, -3.6422498, -0.4766848, 0.4501847
6: -5.8294163, -4.1353111, -5.8272228, -4.1416125, -0.3270991, 0.3255163
7: -2.8035610, -1.2552514, -2.8045852, -1.2577200, -0.4484832, 0.4291295
8: 0.9764191, 1.1554705, 0.9777483, 1.1553980, -0.0161309, 0.0196281
9: -0.1009292, 0.3800510, -0.0997211, 0.3800002, -0.0630268, 0.0611583

Time for backsubstitution: 6.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3081
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 388
type: A, layer: 1, pos: 3080
type: A, layer: 1, pos: 3078
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 2166
type: A, layer: 1, pos: 3077
type: A, layer: 1, pos: 2198
type: A, layer: 1, pos: 373
type: A, layer: 1, pos: 2197
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 2557
type: A, layer: 1, pos: 3082
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 400
type: A, layer: 1, pos: 2375
type: A, layer: 1, pos: 2374
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2809
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 3276
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3489
type: A, layer: 1, pos: 2799
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 3048
type: A, layer: 1, pos: 402
type: A, layer: 1, pos: 2398
type: A, layer: 1, pos: 2366
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 2353
type: A, layer: 1, pos: 2824
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 757
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 403
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 744
type: A, layer: 1, pos: 2117
type: A, layer: 1, pos: 147
type: A, layer: 1, pos: 3063
type: A, layer: 1, pos: 3076
type: A, layer: 1, pos: 2358
type: A, layer: 1, pos: 3021
type: A, layer: 1, pos: 3075
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 2386
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 352
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 2830
type: A, layer: 1, pos: 3046
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 3058
type: A, layer: 1, pos: 3274
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 3032
type: A, layer: 1, pos: 2575
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 820
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3251
type: A, layer: 1, pos: 3262
type: A, layer: 1, pos: 2372
type: A, layer: 1, pos: 2095
type: A, layer: 1, pos: 418
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 3042
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 2362
type: A, layer: 1, pos: 3252
type: A, layer: 1, pos: 2842
type: A, layer: 1, pos: 2361
type: A, layer: 1, pos: 419
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 794
type: A, layer: 1, pos: 2144

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3081

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062932, upper bound: 0.0062089
time: 14.89 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062935, upper bound: 0.0062940
time: 33.63 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 54.90 seconds
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 54.90
Output dim: 8, lower bound: -0.0061500, upper bound: 0.0062763
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 54.90
Output dim: 8, lower bound: -0.0061484, upper bound: 0.0062765
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 54.90
Output dim: 8, lower bound: -0.0061532, upper bound: 0.0062769
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 54.90
Output dim: 8, lower bound: -0.0061547, upper bound: 0.0062958
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 54.90
Output dim: 8, lower bound: -0.0062245, upper bound: 0.0062738
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 54.90
Output dim: 8, lower bound: -0.0062234, upper bound: 0.0062947
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 54.90
Output dim: 8, lower bound: -0.0062930, upper bound: 0.0061890
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 54.90
Output dim: 8, lower bound: -0.0062929, upper bound: 0.0062115
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 54.90
Output dim: 8, lower bound: -0.0062189, upper bound: 0.0062766
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 54.90
Output dim: 8, lower bound: -0.0062192, upper bound: 0.0062932
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 54.90
Output dim: 8, lower bound: -0.0062337, upper bound: 0.0062784
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 54.90
Output dim: 8, lower bound: -0.0062323, upper bound: 0.0062949
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 54.90
Output dim: 8, lower bound: -0.0062924, upper bound: 0.0062136
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 54.90
Output dim: 8, lower bound: -0.0062938, upper bound: 0.0062285
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 54.90
Output dim: 8, lower bound: -0.0062753, upper bound: 0.0062780
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 54.90
Output dim: 8, lower bound: -0.0062749, upper bound: 0.0062920
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 54.90
Output dim: 8, lower bound: -0.0062932, upper bound: 0.0062089
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 54.90
Output dim: 8, lower bound: -0.0062935, upper bound: 0.0062940

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1872873, 0.4362217, -0.1867971, 0.4342444, -0.3194774, 0.3199041
1: -1.4475616, -0.2969993, -1.4440012, -0.2972881, -0.5863627, 0.5841852
2: -3.2341676, -2.2561390, -3.2343440, -2.2614074, -0.1148880, 0.1119721
3: -4.2018633, -2.7073021, -4.1958866, -2.7119727, -0.5038052, 0.4941092
4: -2.8556719, -1.4547261, -2.8548732, -1.4620166, -0.2289496, 0.2288069
5: -5.2484112, -3.6393356, -5.2414045, -3.6440387, -0.4629279, 0.4505094
6: -5.8294473, -4.1355982, -5.8261633, -4.1421204, -0.3215270, 0.3258126
7: -2.7989547, -1.2587160, -2.7993858, -1.2625157, -0.4335890, 0.4303682
8: 0.9783649, 1.1542609, 0.9796731, 1.1545291, -0.0167956, 0.0158754
9: -0.1009065, 0.3793743, -0.0996906, 0.3793412, -0.0623405, 0.0605151

Time for backsubstitution: 6.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 388
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 400
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3251
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2196

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0060908, upper bound: 0.0062927
time: 7.48 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061508, upper bound: 0.0062936
time: 5.10 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1872853, 0.4364690, -0.1868329, 0.4344621, -0.3197173, 0.3199741
1: -1.4476429, -0.2957892, -1.4440017, -0.2961848, -0.5875612, 0.5841613
2: -3.2347980, -2.2530034, -3.2343433, -2.2587049, -0.1181616, 0.1114824
3: -4.2005095, -2.7069638, -4.1958876, -2.7116737, -0.5051665, 0.4942139
4: -2.8557854, -1.4539272, -2.8548737, -1.4613160, -0.2297367, 0.2287315
5: -5.2468123, -3.6389389, -5.2414050, -3.6436906, -0.4642169, 0.4508831
6: -5.8284693, -4.1353121, -5.8261652, -4.1418724, -0.3231620, 0.3255198
7: -2.7989755, -1.2552550, -2.7993863, -1.2592621, -0.4370500, 0.4302027
8: 0.9764191, 1.1547035, 0.9780064, 1.1545291, -0.0164058, 0.0179988
9: -0.1009243, 0.3798881, -0.0996907, 0.3798258, -0.0630459, 0.0603862

Time for backsubstitution: 6.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 388
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 400
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3251
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2196

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061598, upper bound: 0.0062937
time: 3.16 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062219, upper bound: 0.0062936
time: 11.37 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.1873983, 0.4364690, -0.1869669, 0.4345156, -0.3198756, 0.3199210
1: -1.4476576, -0.2956722, -1.4443585, -0.2960563, -0.5875453, 0.5844722
2: -3.2356963, -2.2530024, -3.2353561, -2.2582269, -0.1196490, 0.1109933
3: -4.2016120, -2.7069621, -4.1972141, -2.7112398, -0.5065393, 0.4942665
4: -2.8560815, -1.4539268, -2.8551812, -1.4613428, -0.2311372, 0.2282120
5: -5.2480545, -3.6389384, -5.2429013, -3.6430898, -0.4660471, 0.4505554
6: -5.8284955, -4.1353121, -5.8261805, -4.1417809, -0.3228783, 0.3255502
7: -2.8001709, -1.2552518, -2.8007164, -1.2586788, -0.4387900, 0.4297477
8: 0.9764191, 1.1547904, 0.9779412, 1.1546221, -0.0163592, 0.0181556
9: -0.1009259, 0.3798930, -0.0997012, 0.3798298, -0.0630456, 0.0603892

Time for backsubstitution: 6.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 388
type: B, layer: 1, pos: 3080
type: B, layer: 1, pos: 3078
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 2166
type: B, layer: 1, pos: 3077
type: B, layer: 1, pos: 373
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 398
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 2387
type: B, layer: 1, pos: 2557
type: B, layer: 1, pos: 3082
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 400
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 2375
type: B, layer: 1, pos: 2374
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2809
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 3276
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 3489
type: B, layer: 1, pos: 3081
type: B, layer: 1, pos: 2799
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 3048
type: B, layer: 1, pos: 402
type: B, layer: 1, pos: 2398
type: B, layer: 1, pos: 2366
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 2353
type: B, layer: 1, pos: 2824
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 757
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 403
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 744
type: B, layer: 1, pos: 2117
type: B, layer: 1, pos: 147
type: B, layer: 1, pos: 3063
type: B, layer: 1, pos: 3076
type: B, layer: 1, pos: 2358
type: B, layer: 1, pos: 3021
type: B, layer: 1, pos: 3075
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 2386
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 352
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 2830
type: B, layer: 1, pos: 3046
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 3058
type: B, layer: 1, pos: 3032
type: B, layer: 1, pos: 3274
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 2575
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 820
type: B, layer: 1, pos: 3251
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 3262
type: B, layer: 1, pos: 2372
type: B, layer: 1, pos: 2095
type: B, layer: 1, pos: 418
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 3042
type: B, layer: 1, pos: 2362
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 3252
type: B, layer: 1, pos: 2361
type: B, layer: 1, pos: 2842
type: B, layer: 1, pos: 419
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 794
type: B, layer: 1, pos: 2144

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2196

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062138, upper bound: 0.0062931
time: 17.01 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062222, upper bound: 0.0062919
time: 13.34 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 34.29 + 1775.26 = 1809.54 seconds
