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
execution time: IAR + RelationalAnalysis = 7.82 + 26.48 = 34.30 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0062995, upper bound: 0.0062978

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 398
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 398

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062980, upper bound: 0.0062308
time: 137.66 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062312, upper bound: 0.0062996
time: 11.46 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 149.19 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 149.19
Output dim: 8, lower bound: -0.0062980, upper bound: 0.0062308
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 149.19
Output dim: 8, lower bound: -0.0062312, upper bound: 0.0062996

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3209178, 0.3209309
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5869262, 0.5868973
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1245093, 0.1245512
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5083266, 0.5083369
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2475631, 0.2475872
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4663738, 0.4663778
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3261207, 0.3261400
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4478137, 0.4478137
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199290, 0.0199235
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0620724, 0.0620690

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3021

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062979, upper bound: 0.0062345
time: 3.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062996, upper bound: 0.0062322
time: 101.13 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3209309, 0.3209178
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5868974, 0.5869260
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1245512, 0.1245093
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5083369, 0.5083266
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2475872, 0.2475632
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4663779, 0.4663738
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3261400, 0.3261207
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4478137, 0.4478137
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199235, 0.0199290
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0620690, 0.0620725

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3021
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3021

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062299, upper bound: 0.0063017
time: 4.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062302, upper bound: 0.0063018
time: 3.44 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 14.52 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 14.52
Output dim: 8, lower bound: -0.0062979, upper bound: 0.0062345
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 14.52
Output dim: 8, lower bound: -0.0062996, upper bound: 0.0062322
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 14.52
Output dim: 8, lower bound: -0.0062299, upper bound: 0.0063017
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 14.52
Output dim: 8, lower bound: -0.0062302, upper bound: 0.0063018

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208814, 0.3208944
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5867166, 0.5866879
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1240062, 0.1240469
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5034315, 0.5034407
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2464044, 0.2464244
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4612725, 0.4612767
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3215194, 0.3215301
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4434611, 0.4434687
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199220, 0.0199165
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0619444, 0.0619417

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2347

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062973, upper bound: 0.0062319
time: 8.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062981, upper bound: 0.0062317
time: 3.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208812, 0.3208944
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5867167, 0.5866876
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1240049, 0.1240482
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5034305, 0.5034417
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2464003, 0.2464285
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4612727, 0.4612765
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3215108, 0.3215386
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4434686, 0.4434611
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199219, 0.0199166
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0619452, 0.0619409

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2347

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062974, upper bound: 0.0062312
time: 60.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062988, upper bound: 0.0062326
time: 3.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208944, 0.3208813
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5866877, 0.5867166
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1240482, 0.1240049
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5034418, 0.5034305
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2464285, 0.2464003
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4612766, 0.4612726
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3215386, 0.3215109
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4434611, 0.4434687
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199166, 0.0199219
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0619409, 0.0619452

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2347

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062298, upper bound: 0.0062995
time: 351.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062302, upper bound: 0.0063017
time: 3.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208944, 0.3208814
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5866880, 0.5867165
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1240469, 0.1240062
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5034408, 0.5034314
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2464244, 0.2464044
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4612767, 0.4612725
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3215301, 0.3215194
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4434687, 0.4434611
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199165, 0.0199220
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0619417, 0.0619444

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2347

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062298, upper bound: 0.0063001
time: 18.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062303, upper bound: 0.0062990
time: 3.33 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.08 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.08
Output dim: 8, lower bound: -0.0062973, upper bound: 0.0062319
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.08
Output dim: 8, lower bound: -0.0062981, upper bound: 0.0062317
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.08
Output dim: 8, lower bound: -0.0062974, upper bound: 0.0062312
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.08
Output dim: 8, lower bound: -0.0062988, upper bound: 0.0062326
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.08
Output dim: 8, lower bound: -0.0062298, upper bound: 0.0062995
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.08
Output dim: 8, lower bound: -0.0062302, upper bound: 0.0063017
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.08
Output dim: 8, lower bound: -0.0062298, upper bound: 0.0063001
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.08
Output dim: 8, lower bound: -0.0062303, upper bound: 0.0062990

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208586, 0.3208710
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5866234, 0.5865938
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1236635, 0.1237181
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5015586, 0.5016459
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2455442, 0.2455986
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4593056, 0.4593924
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3195682, 0.3196582
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4420193, 0.4420489
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199213, 0.0199158
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0619050, 0.0619036

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062953, upper bound: 0.0062293
time: 40.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062973, upper bound: 0.0062334
time: 3.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208580, 0.3208717
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5866224, 0.5865948
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1236774, 0.1237042
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5016367, 0.5015678
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2455786, 0.2455642
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4593884, 0.4593098
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3196476, 0.3195789
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4420413, 0.4420269
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199213, 0.0199158
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0619063, 0.0619023

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062974, upper bound: 0.0062309
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062982, upper bound: 0.0062304
time: 3.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208586, 0.3208711
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5866235, 0.5865936
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1236623, 0.1237194
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5015576, 0.5016469
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2455401, 0.2456027
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4593058, 0.4593924
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3195596, 0.3196668
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4420269, 0.4420414
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199212, 0.0199159
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0619058, 0.0619028

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062957, upper bound: 0.0062333
time: 56.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062980, upper bound: 0.0062284
time: 19.77 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208579, 0.3208717
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5866225, 0.5865945
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1236762, 0.1237055
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5016356, 0.5015688
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2455745, 0.2455683
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4593884, 0.4593096
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3196390, 0.3195874
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4420488, 0.4420193
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199212, 0.0199159
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0619071, 0.0619015

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062964, upper bound: 0.0062275
time: 8.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062981, upper bound: 0.0062341
time: 3.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208717, 0.3208579
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5865945, 0.5866225
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1237055, 0.1236762
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5015687, 0.5016357
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2455683, 0.2455745
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4593096, 0.4593883
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3195874, 0.3196389
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4420193, 0.4420488
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199159, 0.0199212
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0619015, 0.0619071

Time for backsubstitution: 6.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062273, upper bound: 0.0063008
time: 8.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062288, upper bound: 0.0062977
time: 18.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208710, 0.3208585
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5865936, 0.5866235
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1237194, 0.1236622
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5016469, 0.5015576
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2456027, 0.2455401
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4593924, 0.4593057
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3196669, 0.3195596
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4420414, 0.4420269
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199159, 0.0199212
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0619028, 0.0619058

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062280, upper bound: 0.0062973
time: 7.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062319, upper bound: 0.0062971
time: 3.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208716, 0.3208579
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5865948, 0.5866224
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1237042, 0.1236774
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5015678, 0.5016367
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2455642, 0.2455786
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4593098, 0.4593884
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3195789, 0.3196476
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4420270, 0.4420413
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199158, 0.0199213
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0619023, 0.0619063

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062281, upper bound: 0.0062982
time: 3.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062294, upper bound: 0.0063010
time: 5.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208710, 0.3208586
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5865938, 0.5866234
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1237181, 0.1236635
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.5016459, 0.5015585
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2455986, 0.2455442
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4593924, 0.4593056
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3196582, 0.3195681
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4420489, 0.4420193
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199158, 0.0199213
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0619036, 0.0619050

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062288, upper bound: 0.0063005
time: 5.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062311, upper bound: 0.0062944
time: 6.79 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 18.47 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 18.47
Output dim: 8, lower bound: -0.0062953, upper bound: 0.0062293
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 18.47
Output dim: 8, lower bound: -0.0062973, upper bound: 0.0062334
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 18.47
Output dim: 8, lower bound: -0.0062974, upper bound: 0.0062309
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 18.47
Output dim: 8, lower bound: -0.0062982, upper bound: 0.0062304
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 18.47
Output dim: 8, lower bound: -0.0062957, upper bound: 0.0062333
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 18.47
Output dim: 8, lower bound: -0.0062980, upper bound: 0.0062284
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 18.47
Output dim: 8, lower bound: -0.0062964, upper bound: 0.0062275
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 18.47
Output dim: 8, lower bound: -0.0062981, upper bound: 0.0062341
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 18.47
Output dim: 8, lower bound: -0.0062273, upper bound: 0.0063008
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 18.47
Output dim: 8, lower bound: -0.0062288, upper bound: 0.0062977
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 18.47
Output dim: 8, lower bound: -0.0062280, upper bound: 0.0062973
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 18.47
Output dim: 8, lower bound: -0.0062319, upper bound: 0.0062971
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 18.47
Output dim: 8, lower bound: -0.0062281, upper bound: 0.0062982
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 18.47
Output dim: 8, lower bound: -0.0062294, upper bound: 0.0063010
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 18.47
Output dim: 8, lower bound: -0.0062288, upper bound: 0.0063005
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 18.47
Output dim: 8, lower bound: -0.0062311, upper bound: 0.0062944

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208314, 0.3208430
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5864698, 0.5864397
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1231316, 0.1231999
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4994447, 0.4996034
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2446375, 0.2447143
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4570179, 0.4571862
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3163705, 0.3166152
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4403903, 0.4404277
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199095, 0.0199041
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0617633, 0.0617688

Time for backsubstitution: 6.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2362

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062926, upper bound: 0.0062304
time: 9.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062959, upper bound: 0.0062275
time: 7.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208306, 0.3208432
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5864691, 0.5864401
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1231441, 0.1231861
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4995010, 0.4995320
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2446587, 0.2446919
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4570813, 0.4571047
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3164947, 0.3164605
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4403914, 0.4404199
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199097, 0.0199040
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0617692, 0.0617619

Time for backsubstitution: 6.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2362

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062942, upper bound: 0.0062328
time: 3.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062970, upper bound: 0.0062281
time: 14.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208301, 0.3208436
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5864686, 0.5864406
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1231455, 0.1231848
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4995229, 0.4995103
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2446719, 0.2446787
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4571006, 0.4570855
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3164499, 0.3165054
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4404123, 0.4403991
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199095, 0.0199041
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0617646, 0.0617666

Time for backsubstitution: 6.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2362

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062937, upper bound: 0.0062283
time: 5.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062966, upper bound: 0.0062304
time: 8.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208300, 0.3208444
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5864682, 0.5864412
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1231592, 0.1231722
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4995943, 0.4994539
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2446944, 0.2446575
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4571822, 0.4570220
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3166046, 0.3163812
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4404202, 0.4403980
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199097, 0.0199040
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0617714, 0.0617606

Time for backsubstitution: 6.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2362

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062952, upper bound: 0.0062326
time: 5.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062979, upper bound: 0.0062295
time: 3.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208314, 0.3208430
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5864699, 0.5864394
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1231303, 0.1232011
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4994437, 0.4996044
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2446334, 0.2447184
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4570180, 0.4571863
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3163619, 0.3166238
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4403979, 0.4404202
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199095, 0.0199042
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0617641, 0.0617679

Time for backsubstitution: 6.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2362

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062934, upper bound: 0.0062340
time: 117.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062954, upper bound: 0.0062291
time: 10.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208306, 0.3208432
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5864692, 0.5864399
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1231428, 0.1231874
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4995001, 0.4995330
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2446546, 0.2446960
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4570814, 0.4571047
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3164862, 0.3164691
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4403991, 0.4404124
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199096, 0.0199041
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0617701, 0.0617611

Time for backsubstitution: 6.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2362

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062944, upper bound: 0.0062301
time: 3.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062969, upper bound: 0.0062291
time: 3.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208300, 0.3208437
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5864688, 0.5864404
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1231442, 0.1231861
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4995218, 0.4995113
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2446678, 0.2446828
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4571006, 0.4570853
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3164413, 0.3165140
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4404199, 0.4403915
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199094, 0.0199042
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0617654, 0.0617658

Time for backsubstitution: 6.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2362

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062933, upper bound: 0.0062306
time: 78.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062962, upper bound: 0.0062285
time: 3.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208299, 0.3208445
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5864683, 0.5864411
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1231579, 0.1231735
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4995932, 0.4994549
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2446902, 0.2446616
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4571822, 0.4570219
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3165959, 0.3163897
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4404277, 0.4403903
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199096, 0.0199041
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0617722, 0.0617598

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2362

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062955, upper bound: 0.0062297
time: 12.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062982, upper bound: 0.0062312
time: 3.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208445, 0.3208299
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5864410, 0.5864684
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1231735, 0.1231579
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4994549, 0.4995931
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2446616, 0.2446902
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4570220, 0.4571822
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3163897, 0.3165959
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4403903, 0.4404277
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199041, 0.0199096
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0617598, 0.0617722

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2362

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062243, upper bound: 0.0063000
time: 9.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062277, upper bound: 0.0062958
time: 11.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208437, 0.3208300
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5864403, 0.5864689
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1231861, 0.1231442
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4995113, 0.4995217
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2446828, 0.2446678
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4570854, 0.4571006
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3165140, 0.3164412
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4403915, 0.4404199
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199042, 0.0199094
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0617658, 0.0617654

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2362

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062265, upper bound: 0.0062987
time: 4.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062292, upper bound: 0.0062961
time: 9.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208432, 0.3208306
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5864398, 0.5864694
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1231874, 0.1231428
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4995330, 0.4995001
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2446960, 0.2446546
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4571047, 0.4570814
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3164692, 0.3164862
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4404124, 0.4403991
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199041, 0.0199096
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0617611, 0.0617701

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2362

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062260, upper bound: 0.0062995
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062285, upper bound: 0.0062978
time: 92.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208430, 0.3208314
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5864393, 0.5864701
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1232011, 0.1231303
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4996044, 0.4994437
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2447184, 0.2446334
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4571863, 0.4570180
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3166238, 0.3163619
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4404202, 0.4403979
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199042, 0.0199095
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0617679, 0.0617641

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2362

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062283, upper bound: 0.0062976
time: 43.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062297, upper bound: 0.0062919
time: 17.17 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208444, 0.3208300
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5864411, 0.5864683
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1231722, 0.1231592
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4994540, 0.4995942
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2446575, 0.2446944
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4570221, 0.4571822
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3163812, 0.3166046
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4403980, 0.4404202
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199040, 0.0199097
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0617606, 0.0617714

Time for backsubstitution: 6.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2362

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062244, upper bound: 0.0062981
time: 31.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062284, upper bound: 0.0062952
time: 15.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208436, 0.3208301
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5864405, 0.5864688
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1231848, 0.1231455
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4995103, 0.4995228
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2446787, 0.2446719
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4570855, 0.4571006
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3165055, 0.3164499
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4403991, 0.4404123
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199041, 0.0199095
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0617666, 0.0617646

Time for backsubstitution: 6.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2362

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062266, upper bound: 0.0063008
time: 4.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062295, upper bound: 0.0062931
time: 13.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208432, 0.3208306
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5864400, 0.5864692
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1231861, 0.1231441
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4995319, 0.4995010
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2446918, 0.2446587
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4571047, 0.4570813
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3164605, 0.3164946
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4404199, 0.4403914
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199040, 0.0199097
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0617619, 0.0617692

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2362

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062260, upper bound: 0.0063007
time: 10.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062288, upper bound: 0.0062974
time: 4.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3208430, 0.3208314
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5864395, 0.5864699
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1231999, 0.1231316
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4996033, 0.4994447
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2447143, 0.2446375
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4571862, 0.4570178
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3166153, 0.3163704
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4404277, 0.4403903
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0199041, 0.0199095
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0617688, 0.0617633

Time for backsubstitution: 6.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2362
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2362

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062277, upper bound: 0.0063006
time: 7.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062307, upper bound: 0.0062966
time: 6.15 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 19.50 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062926, upper bound: 0.0062304
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062959, upper bound: 0.0062275
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062942, upper bound: 0.0062328
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062970, upper bound: 0.0062281
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062937, upper bound: 0.0062283
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062966, upper bound: 0.0062304
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062952, upper bound: 0.0062326
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062979, upper bound: 0.0062295
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062934, upper bound: 0.0062340
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062954, upper bound: 0.0062291
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062944, upper bound: 0.0062301
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062969, upper bound: 0.0062291
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062933, upper bound: 0.0062306
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062962, upper bound: 0.0062285
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062955, upper bound: 0.0062297
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062982, upper bound: 0.0062312
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062243, upper bound: 0.0063000
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062277, upper bound: 0.0062958
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062265, upper bound: 0.0062987
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062292, upper bound: 0.0062961
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062260, upper bound: 0.0062995
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062285, upper bound: 0.0062978
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062283, upper bound: 0.0062976
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062297, upper bound: 0.0062919
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062244, upper bound: 0.0062981
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062284, upper bound: 0.0062952
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062266, upper bound: 0.0063008
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062295, upper bound: 0.0062931
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062260, upper bound: 0.0063007
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062288, upper bound: 0.0062974
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062277, upper bound: 0.0063006
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.50
Output dim: 8, lower bound: -0.0062307, upper bound: 0.0062966

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3207424, 0.3207571
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5862243, 0.5861942
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1214168, 0.1214842
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4937094, 0.4938866
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2403060, 0.2403903
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4509937, 0.4511810
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3096707, 0.3098017
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4381481, 0.4382233
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198921, 0.0198858
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0616993, 0.0617024

Time for backsubstitution: 6.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2361

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062922, upper bound: 0.0062274
time: 11.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062922, upper bound: 0.0062278
time: 7.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3207445, 0.3207542
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5862234, 0.5861946
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1214285, 0.1214684
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4937843, 0.4937884
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2403347, 0.2403552
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4510761, 0.4510712
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3096812, 0.3097436
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4381871, 0.4381748
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198913, 0.0198865
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0617029, 0.0616979

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2361

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062921, upper bound: 0.0062269
time: 4.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062921, upper bound: 0.0062250
time: 8.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3207417, 0.3207571
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5862236, 0.5861945
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1214268, 0.1214705
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4937575, 0.4938152
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2403229, 0.2403678
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4510477, 0.4510995
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3097805, 0.3096470
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4381463, 0.4382155
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198922, 0.0198857
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0617051, 0.0616955

Time for backsubstitution: 6.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2361

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062935, upper bound: 0.0062245
time: 5.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062937, upper bound: 0.0062271
time: 8.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3207441, 0.3207547
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5862229, 0.5861951
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1214299, 0.1214671
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4938061, 0.4937668
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2403479, 0.2403422
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4510954, 0.4510520
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3096364, 0.3097886
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4382080, 0.4381540
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198912, 0.0198867
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0616982, 0.0617024

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2361

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062913, upper bound: 0.0062277
time: 4.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062916, upper bound: 0.0062282
time: 3.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.1878565, 0.4346020, -0.1878565, 0.4346020, -0.3207412, 0.3207576
1: -1.4451120, -0.2953426, -1.4451120, -0.2953426, -0.5862231, 0.5861949
2: -3.2387247, -2.2567830, -3.2387247, -2.2567830, -0.1214280, 0.1214692
3: -4.2010460, -2.7080193, -4.2010460, -2.7080193, -0.4937792, 0.4937936
4: -2.8620744, -1.4602892, -2.8620744, -1.4602892, -0.2403359, 0.2403546
5: -5.2471619, -3.6396251, -5.2471619, -3.6396251, -0.4510671, 0.4510803
6: -5.8273859, -4.1400385, -5.8273859, -4.1400385, -0.3097355, 0.3096920
7: -2.8060415, -1.2545235, -2.8060415, -1.2545235, -0.4381673, 0.4381947
8: 0.9777450, 1.1554776, 0.9777450, 1.1554776, -0.0198921, 0.0198858
9: -0.0997217, 0.3802199, -0.0997217, 0.3802199, -0.0617005, 0.0617002

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2361
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 757
type: DSZ, layer: 1, pos: 2799
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 400
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 744
type: DSZ, layer: 1, pos: 402
type: DSZ, layer: 1, pos: 352
type: DSZ, layer: 1, pos: 3262
type: DSZ, layer: 1, pos: 3489
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2166
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3082
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3276
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 373
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 820
type: DSZ, layer: 1, pos: 388
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 3081
type: DSZ, layer: 1, pos: 403
type: DSZ, layer: 1, pos: 2398
type: DSZ, layer: 1, pos: 418
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 419
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 794
type: DSZ, layer: 1, pos: 2095
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2105
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2144
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2337
type: DSZ, layer: 1, pos: 2353
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2366
type: DSZ, layer: 1, pos: 2372
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2374
type: DSZ, layer: 1, pos: 2375
type: DSZ, layer: 1, pos: 2386
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2575
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2809
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2838
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3032
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3042
type: DSZ, layer: 1, pos: 3046
type: DSZ, layer: 1, pos: 3048
type: DSZ, layer: 1, pos: 3063
type: DSZ, layer: 1, pos: 3075
type: DSZ, layer: 1, pos: 3076
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3078
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3080
type: DSZ, layer: 1, pos: 3251
type: DSZ, layer: 1, pos: 3252
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 3492

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2361

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062921, upper bound: 0.0062261
time: 4.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0062928, upper bound: 0.0062265
time: 7.05 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 17.39 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 17.39
Output dim: 8, lower bound: -0.0062922, upper bound: 0.0062274
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 17.39
Output dim: 8, lower bound: -0.0062922, upper bound: 0.0062278
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 17.39
Output dim: 8, lower bound: -0.0062921, upper bound: 0.0062269
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 17.39
Output dim: 8, lower bound: -0.0062921, upper bound: 0.0062250
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 17.39
Output dim: 8, lower bound: -0.0062935, upper bound: 0.0062245
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 17.39
Output dim: 8, lower bound: -0.0062937, upper bound: 0.0062271
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 17.39
Output dim: 8, lower bound: -0.0062913, upper bound: 0.0062277
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 17.39
Output dim: 8, lower bound: -0.0062916, upper bound: 0.0062282
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 17.39
Output dim: 8, lower bound: -0.0062921, upper bound: 0.0062261
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 17.39
Output dim: 8, lower bound: -0.0062928, upper bound: 0.0062265
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 8, lower bound: -0.0062952, upper bound: 0.0062326
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 8, lower bound: -0.0062979, upper bound: 0.0062295
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 8, lower bound: -0.0062934, upper bound: 0.0062340
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 8, lower bound: -0.0062954, upper bound: 0.0062291
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 8, lower bound: -0.0062944, upper bound: 0.0062301
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 8, lower bound: -0.0062969, upper bound: 0.0062291
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 8, lower bound: -0.0062962, upper bound: 0.0062285
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 8, lower bound: -0.0062955, upper bound: 0.0062297
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 8, lower bound: -0.0062982, upper bound: 0.0062312
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 8, lower bound: -0.0062243, upper bound: 0.0063000
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 8, lower bound: -0.0062277, upper bound: 0.0062958
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 8, lower bound: -0.0062265, upper bound: 0.0062987
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 8, lower bound: -0.0062292, upper bound: 0.0062961
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 8, lower bound: -0.0062260, upper bound: 0.0062995
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 8, lower bound: -0.0062285, upper bound: 0.0062978
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 8, lower bound: -0.0062283, upper bound: 0.0062976
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 8, lower bound: -0.0062244, upper bound: 0.0062981
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 8, lower bound: -0.0062284, upper bound: 0.0062952
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 8, lower bound: -0.0062266, upper bound: 0.0063008
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 8, lower bound: -0.0062260, upper bound: 0.0063007
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 8, lower bound: -0.0062288, upper bound: 0.0062974
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 8, lower bound: -0.0062277, upper bound: 0.0063006
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 8, lower bound: -0.0062307, upper bound: 0.0062966

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 34.30 + 1770.24 = 1804.54 seconds
