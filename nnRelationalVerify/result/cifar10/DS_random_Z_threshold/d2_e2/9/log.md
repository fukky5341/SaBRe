## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 9)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.0761405832


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514421, 0.3514421)
1: (-3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6347626, 0.6347628)
2: (-4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5370792, 0.5370792)
3: (-9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8554106, 0.8554107)
4: (-5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621230, 0.2621230)
5: (-11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9055902, 0.9055902)
6: (-11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3121069, 0.3121069)
7: (-7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634414, 1.2634413)
8: (0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330809, 0.4330808)
9: (-1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5015371, 0.5015372)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 6.93 + 29.73 = 36.65 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0762168, upper bound: 0.0762168

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3443

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2685

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762167, upper bound: 0.0762167
time: 22.76 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762167, upper bound: 0.0762160
time: 232.02 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 254.79 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 254.79
Output dim: 8, lower bound: -0.0762167, upper bound: 0.0762167
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 254.79
Output dim: 8, lower bound: -0.0762167, upper bound: 0.0762160

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514421, 0.3514421
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6347626, 0.6347628
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5370792, 0.5370792
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8554106, 0.8554107
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621230, 0.2621230
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9055902, 0.9055902
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3121069, 0.3121069
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634414, 1.2634413
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330809, 0.4330808
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5015371, 0.5015372

Time for backsubstitution: 5.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762156, upper bound: 0.0762160
time: 11.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762155, upper bound: 0.0762159
time: 14.93 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514421, 0.3514421
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6347626, 0.6347628
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5370792, 0.5370792
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8554106, 0.8554107
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621230, 0.2621230
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9055902, 0.9055902
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3121069, 0.3121069
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634414, 1.2634413
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330809, 0.4330808
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5015371, 0.5015372

Time for backsubstitution: 5.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3467

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0759857, upper bound: 0.0762143
time: 165.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762156, upper bound: 0.0759833
time: 388.04 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 559.68 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 559.68
Output dim: 8, lower bound: -0.0762156, upper bound: 0.0762160
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 559.68
Output dim: 8, lower bound: -0.0762155, upper bound: 0.0762159
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 559.68
Output dim: 8, lower bound: -0.0759857, upper bound: 0.0762143
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 559.68
Output dim: 8, lower bound: -0.0762156, upper bound: 0.0759833

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514389, 0.3514419
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6347286, 0.6347207
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5368425, 0.5368466
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8523152, 0.8523514
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2618023, 0.2618147
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9023250, 0.9023669
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3108529, 0.3108765
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2628545, 1.2628868
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330778, 0.4330777
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5015371, 0.5015376

Time for backsubstitution: 5.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 3497

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3384

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761660, upper bound: 0.0762153
time: 10.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762148, upper bound: 0.0761665
time: 20.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514419, 0.3514389
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6347207, 0.6347286
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5368467, 0.5368425
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8523514, 0.8523152
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2618147, 0.2618023
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9023669, 0.9023250
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3108765, 0.3108530
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2628868, 1.2628543
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330777, 0.4330778
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5015376, 0.5015371

Time for backsubstitution: 5.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2420

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 602

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762171, upper bound: 0.0761912
time: 80.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761936, upper bound: 0.0761923
time: 113.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514395, 0.3514400
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6346155, 0.6345755
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5370051, 0.5369805
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8550357, 0.8548452
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2618788, 0.2618700
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9047552, 0.9045214
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3116987, 0.3116325
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2621261, 1.2618601
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4318451, 0.4320498
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5011296, 0.5010278

Time for backsubstitution: 5.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2465

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2138

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0759836, upper bound: 0.0762139
time: 8.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0759830, upper bound: 0.0762137
time: 269.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514400, 0.3514395
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6345755, 0.6346155
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5369805, 0.5370052
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8548452, 0.8550358
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2618700, 0.2618788
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9045214, 0.9047552
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3116325, 0.3116987
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2618600, 1.2621262
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4320498, 0.4318451
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5010278, 0.5011296

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2466

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762140, upper bound: 0.0759847
time: 9.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762140, upper bound: 0.0759840
time: 166.06 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 181.23 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 181.23
Output dim: 8, lower bound: -0.0761660, upper bound: 0.0762153
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 181.23
Output dim: 8, lower bound: -0.0762148, upper bound: 0.0761665
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 181.23
Output dim: 8, lower bound: -0.0762171, upper bound: 0.0761912
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 181.23
Output dim: 8, lower bound: -0.0761936, upper bound: 0.0761923
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 181.23
Output dim: 8, lower bound: -0.0759836, upper bound: 0.0762139
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 181.23
Output dim: 8, lower bound: -0.0759830, upper bound: 0.0762137
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 181.23
Output dim: 8, lower bound: -0.0762140, upper bound: 0.0759847
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 181.23
Output dim: 8, lower bound: -0.0762140, upper bound: 0.0759840

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514381, 0.3514411
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6347258, 0.6347180
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5368334, 0.5368375
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8523121, 0.8523483
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2618016, 0.2618140
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9023203, 0.9023620
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3108495, 0.3108732
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2628560, 1.2628882
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330794, 0.4330793
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5015306, 0.5015312

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 2602

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2641

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761460, upper bound: 0.0762017
time: 171.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761526, upper bound: 0.0761950
time: 35.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514380, 0.3514411
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6347259, 0.6347179
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5368333, 0.5368376
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8523121, 0.8523483
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2618016, 0.2618140
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9023200, 0.9023622
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3108497, 0.3108730
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2628559, 1.2628882
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330794, 0.4330793
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5015307, 0.5015311

Time for backsubstitution: 5.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2472

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3383

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761407, upper bound: 0.0761658
time: 51.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762148, upper bound: 0.0760928
time: 23.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3510757, 0.3510927
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6336613, 0.6336080
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5358834, 0.5359328
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8517039, 0.8516349
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2603541, 0.2604339
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9014829, 0.9014037
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3102902, 0.3102349
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2628139, 1.2627752
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4329416, 0.4329333
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5006372, 0.5005841

Time for backsubstitution: 5.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 621

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2582

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762060, upper bound: 0.0761820
time: 103.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762062, upper bound: 0.0761816
time: 323.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3510957, 0.3510727
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6336002, 0.6336692
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5359370, 0.5358793
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8516712, 0.8516676
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2604463, 0.2603416
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9014456, 0.9014411
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3102584, 0.3102667
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2628076, 1.2627816
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4329332, 0.4329417
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5005846, 0.5006368

Time for backsubstitution: 5.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2433

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2432

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761813, upper bound: 0.0761766
time: 145.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761530, upper bound: 0.0762051
time: 151.21 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514985, 0.3514987
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6336515, 0.6335569
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5347347, 0.5348534
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8430459, 0.8435805
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2574794, 0.2577187
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.8908656, 0.8914329
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.2999120, 0.3004856
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2565204, 1.2566062
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4318428, 0.4320479
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5010775, 0.5009818

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 346

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2173

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0759768, upper bound: 0.0762116
time: 7.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0759817, upper bound: 0.0759761
time: 100.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514983, 0.3514989
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6335970, 0.6336114
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5348781, 0.5347100
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8437711, 0.8428553
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2577274, 0.2574707
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.8916667, 0.8906318
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3005518, 0.2998459
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2568723, 1.2562542
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4318432, 0.4320475
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5010836, 0.5009757

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3387

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 98

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0759809, upper bound: 0.0762133
time: 95.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0759827, upper bound: 0.0762127
time: 10.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514400, 0.3514395
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6345755, 0.6346155
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5369805, 0.5370052
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8548452, 0.8550358
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2618700, 0.2618788
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9045214, 0.9047552
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3116325, 0.3116987
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2618600, 1.2621262
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4320498, 0.4318451
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5010278, 0.5011296

Time for backsubstitution: 5.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2456

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 621

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762136, upper bound: 0.0759783
time: 167.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762083, upper bound: 0.0759834
time: 174.98 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514400, 0.3514395
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6345755, 0.6346155
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5369805, 0.5370052
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8548452, 0.8550358
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2618700, 0.2618788
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9045214, 0.9047552
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3116325, 0.3116987
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2618600, 1.2621262
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4320498, 0.4318451
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5010278, 0.5011296

Time for backsubstitution: 5.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 370

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3469

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0758936, upper bound: 0.0759837
time: 98.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762134, upper bound: 0.0756636
time: 141.71 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 246.55 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 246.55
Output dim: 8, lower bound: -0.0761460, upper bound: 0.0762017
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 246.55
Output dim: 8, lower bound: -0.0761526, upper bound: 0.0761950
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 246.55
Output dim: 8, lower bound: -0.0761407, upper bound: 0.0761658
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 246.55
Output dim: 8, lower bound: -0.0762148, upper bound: 0.0760928
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 246.55
Output dim: 8, lower bound: -0.0762060, upper bound: 0.0761820
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 246.55
Output dim: 8, lower bound: -0.0762062, upper bound: 0.0761816
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 246.55
Output dim: 8, lower bound: -0.0761813, upper bound: 0.0761766
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 246.55
Output dim: 8, lower bound: -0.0761530, upper bound: 0.0762051
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 246.55
Output dim: 8, lower bound: -0.0759768, upper bound: 0.0762116
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 246.55
Output dim: 8, lower bound: -0.0759817, upper bound: 0.0759761
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 246.55
Output dim: 8, lower bound: -0.0759809, upper bound: 0.0762133
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 246.55
Output dim: 8, lower bound: -0.0759827, upper bound: 0.0762127
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 246.55
Output dim: 8, lower bound: -0.0762136, upper bound: 0.0759783
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 246.55
Output dim: 8, lower bound: -0.0762083, upper bound: 0.0759834
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 246.55
Output dim: 8, lower bound: -0.0758936, upper bound: 0.0759837
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 246.55
Output dim: 8, lower bound: -0.0762134, upper bound: 0.0756636

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514301, 0.3514334
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6344051, 0.6344152
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5368274, 0.5368306
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8523028, 0.8523387
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2617126, 0.2617125
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9022942, 0.9023359
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3106543, 0.3106851
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2628120, 1.2628429
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4329526, 0.4329556
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5010992, 0.5011200

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2464

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761458, upper bound: 0.0762024
time: 15.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761458, upper bound: 0.0762015
time: 84.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514304, 0.3514332
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6344230, 0.6343972
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5368265, 0.5368316
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8523024, 0.8523391
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2617001, 0.2617250
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9022942, 0.9023359
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3106615, 0.3106779
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2628106, 1.2628443
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4329557, 0.4329525
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5011194, 0.5010998

Time for backsubstitution: 5.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 3564
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2470

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2655

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761400, upper bound: 0.0761831
time: 9.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761402, upper bound: 0.0761827
time: 83.81 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 99.70 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 99.70
Output dim: 8, lower bound: -0.0761458, upper bound: 0.0762024
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 99.70
Output dim: 8, lower bound: -0.0761458, upper bound: 0.0762015
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 99.70
Output dim: 8, lower bound: -0.0761400, upper bound: 0.0761831
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 99.70
Output dim: 8, lower bound: -0.0761402, upper bound: 0.0761827
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 99.70
Output dim: 8, lower bound: -0.0761407, upper bound: 0.0761658
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 99.70
Output dim: 8, lower bound: -0.0762148, upper bound: 0.0760928
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 99.70
Output dim: 8, lower bound: -0.0762060, upper bound: 0.0761820
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 99.70
Output dim: 8, lower bound: -0.0762062, upper bound: 0.0761816
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 99.70
Output dim: 8, lower bound: -0.0761813, upper bound: 0.0761766
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 99.70
Output dim: 8, lower bound: -0.0761530, upper bound: 0.0762051
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 99.70
Output dim: 8, lower bound: -0.0759768, upper bound: 0.0762116
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 99.70
Output dim: 8, lower bound: -0.0759809, upper bound: 0.0762133
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 99.70
Output dim: 8, lower bound: -0.0759827, upper bound: 0.0762127
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 99.70
Output dim: 8, lower bound: -0.0762136, upper bound: 0.0759783
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 99.70
Output dim: 8, lower bound: -0.0762083, upper bound: 0.0759834
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 99.70
Output dim: 8, lower bound: -0.0762134, upper bound: 0.0756636

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 36.65 + 3603.76 = 3640.42 seconds
