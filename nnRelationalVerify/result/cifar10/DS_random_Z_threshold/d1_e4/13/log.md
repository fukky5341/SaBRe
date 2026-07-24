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
execution time: IAR + RelationalAnalysis = 7.93 + 27.09 = 35.02 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0062995, upper bound: 0.0062978

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2800

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2104

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062986, upper bound: 0.0062995
time: 3.54 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062988, upper bound: 0.0063003
time: 25.96 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 29.50 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 29.50
Output dim: 8, lower bound: -0.0062986, upper bound: 0.0062995
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 29.50
Output dim: 8, lower bound: -0.0062988, upper bound: 0.0063003

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210423, 0.3210422
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5870066, 0.5870059
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1242523, 0.1242662
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5037073, 0.5038202
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2470470, 0.2470634
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4615213, 0.4616380
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3228942, 0.3229739
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4448419, 0.4449374
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198971, 0.0198972
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0621280, 0.0621282

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2366

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2105

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062942, upper bound: 0.0063027
time: 6.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062975, upper bound: 0.0062958
time: 5.06 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210423, 0.3210423
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5870059, 0.5870066
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1242662, 0.1242524
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5038202, 0.5037073
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2470634, 0.2470470
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4616381, 0.4615213
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3229739, 0.3228942
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4449373, 0.4448418
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198972, 0.0198971
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0621282, 0.0621281

Time for backsubstitution: 6.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2576

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 757

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062972, upper bound: 0.0062995
time: 3.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062982, upper bound: 0.0063027
time: 17.13 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 26.91 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 26.91
Output dim: 8, lower bound: -0.0062942, upper bound: 0.0063027
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 26.91
Output dim: 8, lower bound: -0.0062975, upper bound: 0.0062958
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 26.91
Output dim: 8, lower bound: -0.0062972, upper bound: 0.0062995
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 26.91
Output dim: 8, lower bound: -0.0062982, upper bound: 0.0063027

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210398, 0.3210397
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5869659, 0.5869626
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1241142, 0.1241420
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5026988, 0.5029206
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2468932, 0.2469319
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4604684, 0.4606975
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3221204, 0.3222816
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4442245, 0.4444090
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198826, 0.0198839
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0621260, 0.0621261

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 744

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3276

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062937, upper bound: 0.0062995
time: 3.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062957, upper bound: 0.0062985
time: 22.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210398, 0.3210398
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5869632, 0.5869652
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1241272, 0.1241280
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5028032, 0.5028117
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2469155, 0.2469096
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4605760, 0.4605851
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3221974, 0.3222001
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4443116, 0.4443200
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198835, 0.0198826
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0621260, 0.0621261

Time for backsubstitution: 6.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 400

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062980, upper bound: 0.0062461
time: 11.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062443, upper bound: 0.0062969
time: 14.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210419, 0.3210419
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5869792, 0.5869799
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1241873, 0.1241747
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5033495, 0.5032473
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2470051, 0.2469879
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4611503, 0.4610451
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3223746, 0.3223089
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4446993, 0.4446077
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198872, 0.0198874
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0621281, 0.0621280

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3475

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3046

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062989, upper bound: 0.0063008
time: 15.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062973, upper bound: 0.0063028
time: 4.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210419, 0.3210419
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5869792, 0.5869800
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1241886, 0.1241735
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5033602, 0.5032365
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2470043, 0.2469887
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4611619, 0.4610336
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3223886, 0.3222949
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4447033, 0.4446037
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198875, 0.0198872
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0621281, 0.0621280

Time for backsubstitution: 6.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2347

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062977, upper bound: 0.0062945
time: 87.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062986, upper bound: 0.0063004
time: 31.37 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 125.09 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 125.09
Output dim: 8, lower bound: -0.0062937, upper bound: 0.0062995
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 125.09
Output dim: 8, lower bound: -0.0062957, upper bound: 0.0062985
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 125.09
Output dim: 8, lower bound: -0.0062980, upper bound: 0.0062461
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 125.09
Output dim: 8, lower bound: -0.0062443, upper bound: 0.0062969
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 125.09
Output dim: 8, lower bound: -0.0062989, upper bound: 0.0063008
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 125.09
Output dim: 8, lower bound: -0.0062973, upper bound: 0.0063028
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 125.09
Output dim: 8, lower bound: -0.0062977, upper bound: 0.0062945
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 125.09
Output dim: 8, lower bound: -0.0062986, upper bound: 0.0063004

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210276, 0.3210273
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5869627, 0.5869591
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1240539, 0.1240877
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5025457, 0.5027833
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2467050, 0.2467611
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4602233, 0.4604740
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3220068, 0.3221748
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4440572, 0.4442681
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198680, 0.0198692
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0620821, 0.0620816

Time for backsubstitution: 6.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2373

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2375

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062925, upper bound: 0.0062947
time: 16.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062888, upper bound: 0.0062921
time: 56.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210274, 0.3210276
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5869623, 0.5869594
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1240598, 0.1240817
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5025615, 0.5027676
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2467224, 0.2467436
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4602449, 0.4604525
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3220136, 0.3221680
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4440835, 0.4442418
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198679, 0.0198694
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0620815, 0.0620822

Time for backsubstitution: 6.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 335

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 401

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062931, upper bound: 0.0062619
time: 15.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062614, upper bound: 0.0062985
time: 10.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210397, 0.3210719
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5869420, 0.5868963
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1240379, 0.1241011
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5027557, 0.5027840
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2468378, 0.2468992
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4605346, 0.4605614
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3221932, 0.3222083
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4442974, 0.4443173
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198840, 0.0198805
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0620753, 0.0620472

Time for backsubstitution: 6.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3475

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3058

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062970, upper bound: 0.0062419
time: 3.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062970, upper bound: 0.0062429
time: 3.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210719, 0.3210397
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5868945, 0.5869437
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1241003, 0.1240386
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5027755, 0.5027643
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2469050, 0.2468319
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4605522, 0.4605439
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3222056, 0.3221959
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4443089, 0.4443058
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198813, 0.0198831
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0620471, 0.0620754

Time for backsubstitution: 6.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 779

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2387

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062450, upper bound: 0.0062925
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062446, upper bound: 0.0062928
time: 7.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210376, 0.3210378
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5866129, 0.5866001
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1238526, 0.1238468
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5032384, 0.5031319
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2469981, 0.2469810
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4609977, 0.4608868
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3212125, 0.3211247
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4446957, 0.4446041
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198805, 0.0198806
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0620847, 0.0620841

Time for backsubstitution: 6.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 3021

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062958, upper bound: 0.0063004
time: 37.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062980, upper bound: 0.0062995
time: 3.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210378, 0.3210376
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5865992, 0.5866137
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1238595, 0.1238399
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5032341, 0.5031362
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2469983, 0.2469808
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4609921, 0.4608925
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3211904, 0.3211468
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4446956, 0.4446042
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198804, 0.0198807
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0620842, 0.0620846

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3042

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2842

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062856, upper bound: 0.0062888
time: 9.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062856, upper bound: 0.0062888
time: 9.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210193, 0.3210185
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5868861, 0.5868860
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1238448, 0.1238436
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5014598, 0.5014143
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2461403, 0.2461592
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4591601, 0.4591146
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3204349, 0.3204205
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4432342, 0.4431566
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198868, 0.0198865
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0620888, 0.0620900

Time for backsubstitution: 6.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2362

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2575

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062965, upper bound: 0.0062935
time: 9.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062953, upper bound: 0.0062990
time: 3.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210186, 0.3210192
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5868852, 0.5868868
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1238587, 0.1238297
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5015334, 0.5013361
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2461737, 0.2461248
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4592387, 0.4590318
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3205091, 0.3203411
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4432531, 0.4431345
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198868, 0.0198865
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0620901, 0.0620886

Time for backsubstitution: 6.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2348

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3252

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062990, upper bound: 0.0062983
time: 10.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062988, upper bound: 0.0062982
time: 6.66 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 23.52 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 23.52
Output dim: 8, lower bound: -0.0062925, upper bound: 0.0062947
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 23.52
Output dim: 8, lower bound: -0.0062888, upper bound: 0.0062921
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 23.52
Output dim: 8, lower bound: -0.0062931, upper bound: 0.0062619
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 23.52
Output dim: 8, lower bound: -0.0062614, upper bound: 0.0062985
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 23.52
Output dim: 8, lower bound: -0.0062970, upper bound: 0.0062419
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 23.52
Output dim: 8, lower bound: -0.0062970, upper bound: 0.0062429
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 23.52
Output dim: 8, lower bound: -0.0062450, upper bound: 0.0062925
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 23.52
Output dim: 8, lower bound: -0.0062446, upper bound: 0.0062928
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 23.52
Output dim: 8, lower bound: -0.0062958, upper bound: 0.0063004
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 23.52
Output dim: 8, lower bound: -0.0062980, upper bound: 0.0062995
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 23.52
Output dim: 8, lower bound: -0.0062856, upper bound: 0.0062888
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 23.52
Output dim: 8, lower bound: -0.0062856, upper bound: 0.0062888
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 23.52
Output dim: 8, lower bound: -0.0062965, upper bound: 0.0062935
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 23.52
Output dim: 8, lower bound: -0.0062953, upper bound: 0.0062990
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 23.52
Output dim: 8, lower bound: -0.0062990, upper bound: 0.0062983
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 23.52
Output dim: 8, lower bound: -0.0062988, upper bound: 0.0062982

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3209981, 0.3209955
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5869260, 0.5869212
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1228712, 0.1229022
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4966539, 0.4971769
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2445529, 0.2446597
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4538097, 0.4543955
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3189878, 0.3193097
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4396805, 0.4401284
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198343, 0.0198337
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0620446, 0.0620454

Time for backsubstitution: 6.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3058

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 335

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062916, upper bound: 0.0062748
time: 67.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062729, upper bound: 0.0062935
time: 3.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3205707, 0.3205364
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5870044, 0.5869404
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1235650, 0.1235369
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5024155, 0.5026014
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2458477, 0.2457783
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4599727, 0.4601498
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3220141, 0.3221678
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4438640, 0.4439974
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0197818, 0.0197922
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0615144, 0.0614510

Time for backsubstitution: 6.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3078

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2386

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062615, upper bound: 0.0062970
time: 3.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062608, upper bound: 0.0062934
time: 24.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210381, 0.3210705
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5869270, 0.5868797
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1238786, 0.1239557
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5024251, 0.5024967
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2466583, 0.2467355
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4600618, 0.4601367
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3219131, 0.3219537
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4437817, 0.4438393
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198807, 0.0198770
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0620556, 0.0620257

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 3042

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 403

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062985, upper bound: 0.0062340
time: 13.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062890, upper bound: 0.0062406
time: 24.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210383, 0.3210703
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5869253, 0.5868814
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1238926, 0.1239417
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5024685, 0.5024533
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2466741, 0.2467197
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4601100, 0.4600886
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3219386, 0.3219281
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4438193, 0.4438017
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198805, 0.0198772
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0620538, 0.0620275

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2366

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2362

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062959, upper bound: 0.0062478
time: 3.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062961, upper bound: 0.0062391
time: 13.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210098, 0.3210098
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5864592, 0.5864459
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1232975, 0.1233042
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5010388, 0.5009881
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2459912, 0.2459954
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4586273, 0.4585793
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3178897, 0.3179159
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4430059, 0.4429150
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198681, 0.0198683
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0619425, 0.0619479

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 3075

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2347

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062949, upper bound: 0.0062992
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062950, upper bound: 0.0063005
time: 3.49 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210096, 0.3210100
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5864587, 0.5864464
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1233100, 0.1232918
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5010954, 0.5009323
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2460124, 0.2459742
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4586908, 0.4585165
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3180139, 0.3178019
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4430070, 0.4429143
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198682, 0.0198682
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0619484, 0.0619419

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 3081

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 418

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062968, upper bound: 0.0062916
time: 5.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062873, upper bound: 0.0062948
time: 14.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210173, 0.3210179
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5868765, 0.5868747
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1237745, 0.1237545
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5012482, 0.5011202
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2460521, 0.2460673
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4589155, 0.4587917
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3201756, 0.3201376
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4429213, 0.4428291
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198850, 0.0198843
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0620841, 0.0620841

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062945, upper bound: 0.0062938
time: 16.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062943, upper bound: 0.0062973
time: 121.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210185, 0.3210167
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5868747, 0.5868765
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1237557, 0.1237733
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5011657, 0.5012026
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2460484, 0.2460710
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4588373, 0.4588700
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3201519, 0.3201613
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4429067, 0.4428438
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198846, 0.0198847
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0620828, 0.0620853

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2348

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2602

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062889, upper bound: 0.0062951
time: 8.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062953, upper bound: 0.0062927
time: 3.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210184, 0.3210190
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5868833, 0.5868849
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1238105, 0.1237831
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5014375, 0.5012443
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2461587, 0.2461105
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4591039, 0.4589013
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3204773, 0.3203104
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4431103, 0.4429969
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198868, 0.0198865
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0620791, 0.0620781

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2372

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2374

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062972, upper bound: 0.0062943
time: 18.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062935, upper bound: 0.0062922
time: 10.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210184, 0.3210190
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5868833, 0.5868850
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1238121, 0.1237815
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5014417, 0.5012401
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2461594, 0.2461098
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4591083, 0.4588969
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3204783, 0.3203093
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4431154, 0.4429918
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198868, 0.0198865
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0620796, 0.0620776

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 117

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3063

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062980, upper bound: 0.0062969
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062961, upper bound: 0.0063004
time: 3.55 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 13.27 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 13.27
Output dim: 8, lower bound: -0.0062916, upper bound: 0.0062748
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.27
Output dim: 8, lower bound: -0.0062729, upper bound: 0.0062935
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.27
Output dim: 8, lower bound: -0.0062615, upper bound: 0.0062970
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.27
Output dim: 8, lower bound: -0.0062608, upper bound: 0.0062934
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.27
Output dim: 8, lower bound: -0.0062985, upper bound: 0.0062340
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 13.27
Output dim: 8, lower bound: -0.0062890, upper bound: 0.0062406
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.27
Output dim: 8, lower bound: -0.0062959, upper bound: 0.0062478
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.27
Output dim: 8, lower bound: -0.0062961, upper bound: 0.0062391
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.27
Output dim: 8, lower bound: -0.0062949, upper bound: 0.0062992
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.27
Output dim: 8, lower bound: -0.0062950, upper bound: 0.0063005
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.27
Output dim: 8, lower bound: -0.0062968, upper bound: 0.0062916
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.27
Output dim: 8, lower bound: -0.0062873, upper bound: 0.0062948
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.27
Output dim: 8, lower bound: -0.0062945, upper bound: 0.0062938
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.27
Output dim: 8, lower bound: -0.0062943, upper bound: 0.0062973
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.27
Output dim: 8, lower bound: -0.0062889, upper bound: 0.0062951
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.27
Output dim: 8, lower bound: -0.0062953, upper bound: 0.0062927
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.27
Output dim: 8, lower bound: -0.0062972, upper bound: 0.0062943
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.27
Output dim: 8, lower bound: -0.0062935, upper bound: 0.0062922
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.27
Output dim: 8, lower bound: -0.0062980, upper bound: 0.0062969
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.27
Output dim: 8, lower bound: -0.0062961, upper bound: 0.0063004

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208407, 0.3208508
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5868752, 0.5868651
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1208541, 0.1207423
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4950801, 0.4954703
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2421161, 0.2420498
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4525690, 0.4530520
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3156844, 0.3157634
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4384584, 0.4389147
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0194791, 0.0194987
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0618760, 0.0618716

Time for backsubstitution: 6.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2387

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2828

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062727, upper bound: 0.0062952
time: 13.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062724, upper bound: 0.0062969
time: 3.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3205687, 0.3205347
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5857636, 0.5856986
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1229406, 0.1229164
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5021544, 0.5023346
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2458454, 0.2457753
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4593816, 0.4595522
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3192847, 0.3194229
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4438463, 0.4439802
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0197663, 0.0197766
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0613678, 0.0613054

Time for backsubstitution: 6.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2095

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062511, upper bound: 0.0062846
time: 9.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062511, upper bound: 0.0062846
time: 9.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3205689, 0.3205344
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5857627, 0.5856994
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1229445, 0.1229126
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5021486, 0.5023402
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2458447, 0.2457760
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4593751, 0.4595587
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3192692, 0.3194383
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4438468, 0.4439797
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0197662, 0.0197767
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0613688, 0.0613044

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 388

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2347

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062594, upper bound: 0.0062889
time: 11.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062606, upper bound: 0.0062963
time: 14.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3198681, 0.3198878
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5845958, 0.5846071
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1232571, 0.1233486
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5023746, 0.5024704
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2415330, 0.2415980
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4595179, 0.4596244
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3202655, 0.3202876
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4417132, 0.4418263
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198752, 0.0198711
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0600653, 0.0600852

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3033

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062955, upper bound: 0.0062337
time: 9.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062950, upper bound: 0.0062299
time: 7.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3209523, 0.3209813
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5866799, 0.5866358
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1221972, 0.1222472
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4968123, 0.4967786
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2424550, 0.2424932
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4541661, 0.4541259
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3152181, 0.3153216
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4417116, 0.4416561
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198623, 0.0198599
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0619874, 0.0619636

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2348

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 418

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062949, upper bound: 0.0062312
time: 16.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062853, upper bound: 0.0062486
time: 3.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3209493, 0.3209843
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5866796, 0.5866359
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1221981, 0.1222464
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4967938, 0.4967971
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2424476, 0.2425007
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4541473, 0.4541448
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3153319, 0.3152077
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4416738, 0.4416939
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198631, 0.0198590
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0619898, 0.0619612

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2095

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2372

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062963, upper bound: 0.0062254
time: 6.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062863, upper bound: 0.0062377
time: 3.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3209877, 0.3209865
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5863662, 0.5863519
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1229699, 0.1229917
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4992097, 0.4992399
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2452096, 0.2452478
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4567006, 0.4567407
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3160091, 0.3161391
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4415918, 0.4415236
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198675, 0.0198677
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0619036, 0.0619112

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 418

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062939, upper bound: 0.0062950
time: 4.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062933, upper bound: 0.0062978
time: 5.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3209865, 0.3209872
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5863652, 0.5863528
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1229838, 0.1229766
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4992879, 0.4991590
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2452441, 0.2452138
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4567834, 0.4566525
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3160886, 0.3160353
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4416139, 0.4415009
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198675, 0.0198677
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0619049, 0.0619090

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 352

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2197

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062935, upper bound: 0.0062401
time: 28.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062372, upper bound: 0.0062915
time: 16.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210083, 0.3210089
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5864517, 0.5864389
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1233079, 0.1232899
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5010931, 0.5009304
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2459958, 0.2459596
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4586897, 0.4585157
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3179974, 0.3177848
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4429886, 0.4428980
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198677, 0.0198676
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0619334, 0.0619288

Time for backsubstitution: 6.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 352

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2362

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062950, upper bound: 0.0062828
time: 9.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062976, upper bound: 0.0062841
time: 274.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210084, 0.3210087
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5864512, 0.5864391
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1233082, 0.1232896
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5010934, 0.5009301
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2459979, 0.2459576
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4586899, 0.4585154
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3179968, 0.3177853
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4429908, 0.4428959
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198677, 0.0198677
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0619354, 0.0619268

Time for backsubstitution: 6.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 400

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2372

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062881, upper bound: 0.0062883
time: 4.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062753, upper bound: 0.0063000
time: 3.20 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210006, 0.3210016
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5865437, 0.5865507
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1234241, 0.1234057
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4997625, 0.4996408
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2461360, 0.2461518
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4572839, 0.4571670
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3178313, 0.3178045
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4435043, 0.4434150
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198442, 0.0198425
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0620067, 0.0620071

Time for backsubstitution: 6.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2349

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2638

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062909, upper bound: 0.0062915
time: 6.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062928, upper bound: 0.0062912
time: 13.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210010, 0.3210013
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5865523, 0.5865420
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1234256, 0.1234041
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4997687, 0.4996346
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2461365, 0.2461512
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4572909, 0.4571601
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3178405, 0.3177933
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4435074, 0.4434119
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198432, 0.0198436
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0620071, 0.0620066

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2348

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 335

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062940, upper bound: 0.0062755
time: 12.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062749, upper bound: 0.0062929
time: 5.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210180, 0.3210161
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5867363, 0.5867364
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1215670, 0.1215479
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4958535, 0.4958408
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2397134, 0.2397014
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4529298, 0.4529115
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3140869, 0.3140724
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4388711, 0.4387614
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0197407, 0.0197459
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0619247, 0.0619255

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 821

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2387

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062881, upper bound: 0.0062942
time: 24.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062874, upper bound: 0.0062974
time: 26.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210180, 0.3210161
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5867345, 0.5867382
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1215304, 0.1215845
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4958039, 0.4958905
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2396789, 0.2397359
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4528787, 0.4529626
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3140631, 0.3140962
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4388243, 0.4388082
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0197459, 0.0197408
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0619230, 0.0619272

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3274

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062946, upper bound: 0.0062894
time: 6.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062930, upper bound: 0.0062919
time: 4.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3209651, 0.3209654
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5867336, 0.5867333
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1211329, 0.1211247
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4886312, 0.4886083
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2412268, 0.2412156
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4448544, 0.4449135
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3136266, 0.3135456
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4327711, 0.4328716
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198009, 0.0197988
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0620578, 0.0620571

Time for backsubstitution: 6.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3032

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 779

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062980, upper bound: 0.0062948
time: 3.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062980, upper bound: 0.0062903
time: 4.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3209647, 0.3209657
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5867320, 0.5867352
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1211520, 0.1211055
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4888014, 0.4884380
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2412639, 0.2411785
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4451160, 0.4446518
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3137125, 0.3134596
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4329850, 0.4326577
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0197990, 0.0198006
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0620581, 0.0620568

Time for backsubstitution: 6.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 3080

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3489

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062798, upper bound: 0.0062990
time: 3.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062928, upper bound: 0.0062807
time: 14.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210156, 0.3210164
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5869573, 0.5869602
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1219764, 0.1220274
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4942083, 0.4943175
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2444956, 0.2445256
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4510658, 0.4512057
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3162780, 0.3162380
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4376197, 0.4377939
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198371, 0.0198342
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0620978, 0.0620964

Time for backsubstitution: 6.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3019

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3046

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062962, upper bound: 0.0062891
time: 5.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062958, upper bound: 0.0062965
time: 3.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3210158, 0.3210163
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5869584, 0.5869590
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1220580, 0.1219458
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4945191, 0.4940066
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2445752, 0.2444461
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4514171, 0.4508544
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3164070, 0.3161090
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4379176, 0.4374960
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198345, 0.0198368
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0620983, 0.0620959

Time for backsubstitution: 6.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 3075

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 820

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062943, upper bound: 0.0062952
time: 4.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062939, upper bound: 0.0062961
time: 5.73 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 16.93 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062727, upper bound: 0.0062952
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062724, upper bound: 0.0062969
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062511, upper bound: 0.0062846
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062511, upper bound: 0.0062846
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062594, upper bound: 0.0062889
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062606, upper bound: 0.0062963
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062955, upper bound: 0.0062337
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062950, upper bound: 0.0062299
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062949, upper bound: 0.0062312
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062853, upper bound: 0.0062486
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062963, upper bound: 0.0062254
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062863, upper bound: 0.0062377
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062939, upper bound: 0.0062950
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062933, upper bound: 0.0062978
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062935, upper bound: 0.0062401
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062372, upper bound: 0.0062915
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062950, upper bound: 0.0062828
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062976, upper bound: 0.0062841
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062881, upper bound: 0.0062883
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062753, upper bound: 0.0063000
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062909, upper bound: 0.0062915
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062928, upper bound: 0.0062912
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062940, upper bound: 0.0062755
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062749, upper bound: 0.0062929
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062881, upper bound: 0.0062942
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062874, upper bound: 0.0062974
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062946, upper bound: 0.0062894
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062930, upper bound: 0.0062919
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062980, upper bound: 0.0062948
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062980, upper bound: 0.0062903
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062798, upper bound: 0.0062990
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062928, upper bound: 0.0062807
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062962, upper bound: 0.0062891
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062958, upper bound: 0.0062965
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062943, upper bound: 0.0062952
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 16.93
Output dim: 8, lower bound: -0.0062939, upper bound: 0.0062961

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208369, 0.3208466
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5868444, 0.5868320
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1206867, 0.1205877
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4945466, 0.4949515
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2417219, 0.2416674
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4519971, 0.4524960
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3151994, 0.3153020
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4381968, 0.4386528
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0194767, 0.0194959
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0618757, 0.0618713

Time for backsubstitution: 6.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 401

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062710, upper bound: 0.0062605
time: 16.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062410, upper bound: 0.0062913
time: 10.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208365, 0.3208470
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5868421, 0.5868344
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1206996, 0.1205749
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4945614, 0.4949368
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2417337, 0.2416555
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4520131, 0.4524800
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3152230, 0.3152784
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4381965, 0.4386531
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0194763, 0.0194963
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0618757, 0.0618713

Time for backsubstitution: 6.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2576

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 779

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062720, upper bound: 0.0062707
time: 58.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062720, upper bound: 0.0062919
time: 31.52 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 96.49 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 96.49
Output dim: 8, lower bound: -0.0062710, upper bound: 0.0062605
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 96.49
Output dim: 8, lower bound: -0.0062410, upper bound: 0.0062913
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 96.49
Output dim: 8, lower bound: -0.0062720, upper bound: 0.0062707
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 96.49
Output dim: 8, lower bound: -0.0062720, upper bound: 0.0062919
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 96.49
Output dim: 8, lower bound: -0.0062606, upper bound: 0.0062963
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 96.49
Output dim: 8, lower bound: -0.0062955, upper bound: 0.0062337
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 96.49
Output dim: 8, lower bound: -0.0062950, upper bound: 0.0062299
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 96.49
Output dim: 8, lower bound: -0.0062949, upper bound: 0.0062312
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 96.49
Output dim: 8, lower bound: -0.0062963, upper bound: 0.0062254
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 96.49
Output dim: 8, lower bound: -0.0062939, upper bound: 0.0062950
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 96.49
Output dim: 8, lower bound: -0.0062933, upper bound: 0.0062978
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 96.49
Output dim: 8, lower bound: -0.0062935, upper bound: 0.0062401
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 96.49
Output dim: 8, lower bound: -0.0062950, upper bound: 0.0062828
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 96.49
Output dim: 8, lower bound: -0.0062976, upper bound: 0.0062841
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 96.49
Output dim: 8, lower bound: -0.0062753, upper bound: 0.0063000
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 96.49
Output dim: 8, lower bound: -0.0062940, upper bound: 0.0062755
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 96.49
Output dim: 8, lower bound: -0.0062881, upper bound: 0.0062942
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 96.49
Output dim: 8, lower bound: -0.0062874, upper bound: 0.0062974
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 96.49
Output dim: 8, lower bound: -0.0062946, upper bound: 0.0062894
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 96.49
Output dim: 8, lower bound: -0.0062980, upper bound: 0.0062948
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 96.49
Output dim: 8, lower bound: -0.0062980, upper bound: 0.0062903
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 96.49
Output dim: 8, lower bound: -0.0062798, upper bound: 0.0062990
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 96.49
Output dim: 8, lower bound: -0.0062962, upper bound: 0.0062891
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 96.49
Output dim: 8, lower bound: -0.0062958, upper bound: 0.0062965
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 96.49
Output dim: 8, lower bound: -0.0062943, upper bound: 0.0062952
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 96.49
Output dim: 8, lower bound: -0.0062939, upper bound: 0.0062961

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 35.02 + 1825.71 = 1860.73 seconds
