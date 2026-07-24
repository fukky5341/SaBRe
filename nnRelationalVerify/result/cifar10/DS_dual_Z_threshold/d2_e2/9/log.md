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
execution time: IAR + RelationalAnalysis = 7.76 + 30.11 = 37.87 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0762168, upper bound: 0.0762168

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3428

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0759545, upper bound: 0.0762165
time: 72.93 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762162, upper bound: 0.0759543
time: 251.91 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 324.91 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 324.91
Output dim: 8, lower bound: -0.0759545, upper bound: 0.0762165
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 324.91
Output dim: 8, lower bound: -0.0762162, upper bound: 0.0759543

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514394, 0.3514394
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6347587, 0.6347586
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5370716, 0.5370721
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8554273, 0.8554280
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621316, 0.2621316
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9056107, 0.9056119
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3121355, 0.3121354
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634376, 1.2634376
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330764, 0.4330766
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5015576, 0.5015590

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3443

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0758331, upper bound: 0.0762143
time: 10.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0759391, upper bound: 0.0760631
time: 9.34 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514395, 0.3514394
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6347586, 0.6347587
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5370720, 0.5370717
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8554279, 0.8554273
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621316, 0.2621316
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9056119, 0.9056107
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3121354, 0.3121355
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634376, 1.2634376
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330766, 0.4330764
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5015590, 0.5015577

Time for backsubstitution: 5.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3443

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0760623, upper bound: 0.0759391
time: 25.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762141, upper bound: 0.0758326
time: 42.44 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 73.47 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 73.47
Output dim: 8, lower bound: -0.0758331, upper bound: 0.0762143
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 73.47
Output dim: 8, lower bound: -0.0759391, upper bound: 0.0760631
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 73.47
Output dim: 8, lower bound: -0.0760623, upper bound: 0.0759391
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 73.47
Output dim: 8, lower bound: -0.0762141, upper bound: 0.0758326

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514358, 0.3514358
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6347145, 0.6347121
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5370718, 0.5370720
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8554121, 0.8554131
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621151, 0.2621155
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9055856, 0.9055874
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3121576, 0.3121571
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634324, 1.2634327
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330769, 0.4330771
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5015461, 0.5015475

Time for backsubstitution: 6.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3399

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0756705, upper bound: 0.0757768
time: 67.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0758285, upper bound: 0.0760510
time: 11.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514358, 0.3514358
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6347121, 0.6347145
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5370720, 0.5370718
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8554131, 0.8554122
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621155, 0.2621151
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9055874, 0.9055855
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3121571, 0.3121576
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634327, 1.2634325
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330771, 0.4330769
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5015475, 0.5015461

Time for backsubstitution: 6.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3399
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3399

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0760515, upper bound: 0.0758288
time: 15.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762095, upper bound: 0.0756708
time: 6.13 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.22 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 28.22
Output dim: 8, lower bound: -0.0756705, upper bound: 0.0757768
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 28.22
Output dim: 8, lower bound: -0.0758285, upper bound: 0.0760510
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 28.22
Output dim: 8, lower bound: -0.0760515, upper bound: 0.0758288
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.22
Output dim: 8, lower bound: -0.0762095, upper bound: 0.0756708

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514348, 0.3514348
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6347092, 0.6347116
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5370625, 0.5370624
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8554127, 0.8554118
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621159, 0.2621154
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9055862, 0.9055844
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3121576, 0.3121582
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634344, 1.2634342
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330783, 0.4330781
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5015440, 0.5015426

Time for backsubstitution: 6.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 531
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 531

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0761256, upper bound: 0.0755853
time: 100.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762059, upper bound: 0.0755861
time: 88.97 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 195.91 seconds
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 195.91
Output dim: 8, lower bound: -0.0761256, upper bound: 0.0755853
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 195.91
Output dim: 8, lower bound: -0.0762059, upper bound: 0.0755861

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514347, 0.3514347
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6347089, 0.6347114
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5370566, 0.5370572
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8554070, 0.8554060
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621154, 0.2621148
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9055755, 0.9055736
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3121466, 0.3121461
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634306, 1.2634306
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330780, 0.4330777
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5015384, 0.5015377

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3458
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3458

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761803, upper bound: 0.0755828
time: 9.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762021, upper bound: 0.0755626
time: 108.16 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 124.22 seconds
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 124.22
Output dim: 8, lower bound: -0.0761803, upper bound: 0.0755828
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 124.22
Output dim: 8, lower bound: -0.0762021, upper bound: 0.0755626

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514333, 0.3514333
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6346714, 0.6346721
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5370464, 0.5370469
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8553772, 0.8553764
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621109, 0.2621104
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9055399, 0.9055378
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3121268, 0.3121259
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634287, 1.2634287
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330782, 0.4330780
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5015274, 0.5015268

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 550

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761678, upper bound: 0.0755743
time: 11.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761729, upper bound: 0.0755689
time: 101.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514333, 0.3514333
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6346695, 0.6346740
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5370463, 0.5370471
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8553773, 0.8553763
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621109, 0.2621104
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9055398, 0.9055380
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3121264, 0.3121263
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634287, 1.2634287
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330782, 0.4330780
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5015274, 0.5015268

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 550

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761896, upper bound: 0.0755553
time: 8.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761945, upper bound: 0.0755500
time: 7.31 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 22.28 seconds
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 22.28
Output dim: 8, lower bound: -0.0761678, upper bound: 0.0755743
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 22.28
Output dim: 8, lower bound: -0.0761729, upper bound: 0.0755689
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 22.28
Output dim: 8, lower bound: -0.0761896, upper bound: 0.0755553
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 22.28
Output dim: 8, lower bound: -0.0761945, upper bound: 0.0755500

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514340, 0.3514333
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6346716, 0.6346719
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5370464, 0.5370469
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8553771, 0.8553776
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621098, 0.2621075
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9055399, 0.9055383
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3121257, 0.3121213
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634284, 1.2634277
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330780, 0.4330778
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5015270, 0.5015254

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3383

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0760934, upper bound: 0.0755548
time: 172.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761674, upper bound: 0.0755013
time: 8.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514333, 0.3514333
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6346713, 0.6346721
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5370464, 0.5370469
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8553772, 0.8553763
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621079, 0.2621104
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9055399, 0.9055378
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3121222, 0.3121259
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634279, 1.2634287
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330782, 0.4330778
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5015260, 0.5015268

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 3383

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0760985, upper bound: 0.0755691
time: 34.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761725, upper bound: 0.0754946
time: 50.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514340, 0.3514333
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6346697, 0.6346739
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5370462, 0.5370470
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8553773, 0.8553774
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621099, 0.2621074
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9055398, 0.9055384
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3121252, 0.3121217
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634283, 1.2634279
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330780, 0.4330778
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5015270, 0.5015254

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3383

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0761152, upper bound: 0.0755547
time: 87.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761892, upper bound: 0.0754806
time: 12.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514333, 0.3514333
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6346693, 0.6346740
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5370463, 0.5370470
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8553773, 0.8553762
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621080, 0.2621104
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9055398, 0.9055379
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3121218, 0.3121263
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634277, 1.2634287
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330782, 0.4330778
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5015260, 0.5015268

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3383
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3383

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0761203, upper bound: 0.0754753
time: 107.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761943, upper bound: 0.0754752
time: 96.67 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 211.12 seconds
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 211.12
Output dim: 8, lower bound: -0.0760934, upper bound: 0.0755548
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 211.12
Output dim: 8, lower bound: -0.0761674, upper bound: 0.0755013
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 211.12
Output dim: 8, lower bound: -0.0760985, upper bound: 0.0755691
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 211.12
Output dim: 8, lower bound: -0.0761725, upper bound: 0.0754946
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 211.12
Output dim: 8, lower bound: -0.0761152, upper bound: 0.0755547
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 211.12
Output dim: 8, lower bound: -0.0761892, upper bound: 0.0754806
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 211.12
Output dim: 8, lower bound: -0.0761203, upper bound: 0.0754753
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 211.12
Output dim: 8, lower bound: -0.0761943, upper bound: 0.0754752

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514332, 0.3514325
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6346691, 0.6346695
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5370373, 0.5370378
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8553764, 0.8553768
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621099, 0.2621076
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9055380, 0.9055364
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3121252, 0.3121208
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634299, 1.2634292
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330793, 0.4330792
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5015225, 0.5015210

Time for backsubstitution: 6.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 581

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761668, upper bound: 0.0754758
time: 108.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761624, upper bound: 0.0755007
time: 4.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514324, 0.3514325
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6346688, 0.6346696
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5370374, 0.5370378
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8553765, 0.8553755
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621080, 0.2621105
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9055380, 0.9055359
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3121217, 0.3121254
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634293, 1.2634301
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330796, 0.4330791
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5015216, 0.5015224

Time for backsubstitution: 6.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 581

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761719, upper bound: 0.0754913
time: 5.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761674, upper bound: 0.0754960
time: 4.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514332, 0.3514325
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6346672, 0.6346714
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5370371, 0.5370380
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8553765, 0.8553767
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621100, 0.2621075
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9055379, 0.9055366
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3121247, 0.3121212
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634299, 1.2634293
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330793, 0.4330792
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5015225, 0.5015210

Time for backsubstitution: 6.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 581

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761886, upper bound: 0.0754759
time: 15.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761842, upper bound: 0.0754804
time: 56.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3514325, 0.3514325
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6346669, 0.6346715
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5370371, 0.5370380
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8553765, 0.8553755
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621081, 0.2621104
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9055379, 0.9055361
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3121213, 0.3121258
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634293, 1.2634301
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330796, 0.4330791
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5015215, 0.5015224

Time for backsubstitution: 6.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 581

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761937, upper bound: 0.0754714
time: 7.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761892, upper bound: 0.0754749
time: 128.36 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 143.16 seconds
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 143.16
Output dim: 8, lower bound: -0.0761668, upper bound: 0.0754758
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 143.16
Output dim: 8, lower bound: -0.0761624, upper bound: 0.0755007
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 143.16
Output dim: 8, lower bound: -0.0761719, upper bound: 0.0754913
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 143.16
Output dim: 8, lower bound: -0.0761674, upper bound: 0.0754960
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 143.16
Output dim: 8, lower bound: -0.0761886, upper bound: 0.0754759
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 143.16
Output dim: 8, lower bound: -0.0761842, upper bound: 0.0754804
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 143.16
Output dim: 8, lower bound: -0.0761937, upper bound: 0.0754714
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 143.16
Output dim: 8, lower bound: -0.0761892, upper bound: 0.0754749

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3511536, 0.3511697
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6345018, 0.6344925
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5367593, 0.5367805
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8551099, 0.8551039
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621918, 0.2621915
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9053963, 0.9053909
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3116852, 0.3116701
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634690, 1.2634597
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330891, 0.4330859
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5013116, 0.5012969

Time for backsubstitution: 6.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 459

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0760276, upper bound: 0.0754861
time: 54.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761668, upper bound: 0.0754532
time: 12.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3511704, 0.3511530
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6344922, 0.6345021
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5367799, 0.5367599
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8551035, 0.8551103
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621939, 0.2621894
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9053924, 0.9053948
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3116745, 0.3116808
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634602, 1.2634685
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330862, 0.4330890
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5012985, 0.5013100

Time for backsubstitution: 6.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 459

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0760236, upper bound: 0.0754918
time: 5.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761627, upper bound: 0.0754538
time: 45.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3511529, 0.3511697
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6345015, 0.6344926
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5367593, 0.5367805
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8551100, 0.8551027
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621898, 0.2621945
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9053965, 0.9053903
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3116818, 0.3116748
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634685, 1.2634606
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330894, 0.4330859
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5013106, 0.5012983

Time for backsubstitution: 6.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 459

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0760327, upper bound: 0.0754812
time: 95.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761718, upper bound: 0.0754482
time: 64.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3511696, 0.3511529
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6344919, 0.6345022
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5367799, 0.5367599
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8551035, 0.8551091
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621920, 0.2621924
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9053925, 0.9053943
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3116711, 0.3116854
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634597, 1.2634695
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330864, 0.4330889
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5012975, 0.5013114

Time for backsubstitution: 6.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 459

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0760286, upper bound: 0.0754609
time: 143.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761674, upper bound: 0.0754534
time: 5.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3511536, 0.3511697
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6344998, 0.6344944
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5367591, 0.5367807
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8551100, 0.8551038
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621919, 0.2621915
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9053962, 0.9053910
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3116848, 0.3116706
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634690, 1.2634597
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330891, 0.4330859
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5013115, 0.5012969

Time for backsubstitution: 6.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 459

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0760494, upper bound: 0.0754333
time: 69.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761886, upper bound: 0.0754341
time: 76.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3511704, 0.3511530
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6344903, 0.6345040
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5367798, 0.5367600
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8551036, 0.8551103
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621940, 0.2621893
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9053923, 0.9053950
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3116741, 0.3116813
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634602, 1.2634685
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330862, 0.4330890
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5012984, 0.5013101

Time for backsubstitution: 6.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 459

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0760452, upper bound: 0.0754664
time: 49.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761844, upper bound: 0.0754374
time: 114.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3511529, 0.3511697
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6344995, 0.6344945
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5367592, 0.5367806
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8551100, 0.8551025
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621899, 0.2621944
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9053963, 0.9053905
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3116813, 0.3116752
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634685, 1.2634606
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330894, 0.4330859
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5013106, 0.5012984

Time for backsubstitution: 6.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 459

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0760545, upper bound: 0.0754607
time: 99.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761936, upper bound: 0.0754280
time: 42.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3511696, 0.3511529
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6344900, 0.6345041
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5367798, 0.5367600
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8551036, 0.8551090
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621921, 0.2621922
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9053923, 0.9053944
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3116707, 0.3116859
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634597, 1.2634695
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330864, 0.4330889
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5012975, 0.5013115

Time for backsubstitution: 6.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 459

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0760500, upper bound: 0.0754655
time: 86.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761891, upper bound: 0.0754344
time: 4.94 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 97.72 seconds
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 97.72
Output dim: 8, lower bound: -0.0760276, upper bound: 0.0754861
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 97.72
Output dim: 8, lower bound: -0.0761668, upper bound: 0.0754532
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 97.72
Output dim: 8, lower bound: -0.0760236, upper bound: 0.0754918
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 97.72
Output dim: 8, lower bound: -0.0761627, upper bound: 0.0754538
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 97.72
Output dim: 8, lower bound: -0.0760327, upper bound: 0.0754812
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 97.72
Output dim: 8, lower bound: -0.0761718, upper bound: 0.0754482
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 97.72
Output dim: 8, lower bound: -0.0760286, upper bound: 0.0754609
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 97.72
Output dim: 8, lower bound: -0.0761674, upper bound: 0.0754534
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 97.72
Output dim: 8, lower bound: -0.0760494, upper bound: 0.0754333
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 97.72
Output dim: 8, lower bound: -0.0761886, upper bound: 0.0754341
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 97.72
Output dim: 8, lower bound: -0.0760452, upper bound: 0.0754664
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 97.72
Output dim: 8, lower bound: -0.0761844, upper bound: 0.0754374
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 97.72
Output dim: 8, lower bound: -0.0760545, upper bound: 0.0754607
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 97.72
Output dim: 8, lower bound: -0.0761936, upper bound: 0.0754280
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 97.72
Output dim: 8, lower bound: -0.0760500, upper bound: 0.0754655
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 97.72
Output dim: 8, lower bound: -0.0761891, upper bound: 0.0754344

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3511526, 0.3511688
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6344992, 0.6344900
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5367517, 0.5367728
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8551096, 0.8551036
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621924, 0.2621922
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9053953, 0.9053898
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3116857, 0.3116706
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634703, 1.2634609
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330900, 0.4330868
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5013087, 0.5012939

Time for backsubstitution: 6.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3385

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0761197, upper bound: 0.0754544
time: 5.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761668, upper bound: 0.0754070
time: 60.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3511694, 0.3511520
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6344897, 0.6344995
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5367723, 0.5367522
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8551032, 0.8551100
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621946, 0.2621900
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9053913, 0.9053938
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3116750, 0.3116813
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634615, 1.2634697
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330870, 0.4330899
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5012955, 0.5013071

Time for backsubstitution: 6.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3385

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0761153, upper bound: 0.0754575
time: 48.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761623, upper bound: 0.0754112
time: 133.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3511519, 0.3511688
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6344990, 0.6344901
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5367517, 0.5367728
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8551097, 0.8551023
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621905, 0.2621951
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9053954, 0.9053893
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3116822, 0.3116752
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634698, 1.2634617
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330902, 0.4330868
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5013077, 0.5012953

Time for backsubstitution: 6.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3385

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0761247, upper bound: 0.0754492
time: 5.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761718, upper bound: 0.0754021
time: 8.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3511687, 0.3511520
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6344894, 0.6344996
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5367724, 0.5367522
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8551033, 0.8551088
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621926, 0.2621930
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9053915, 0.9053932
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3116716, 0.3116859
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634610, 1.2634705
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330873, 0.4330898
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5012946, 0.5013084

Time for backsubstitution: 6.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3385

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0761203, upper bound: 0.0754527
time: 6.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761674, upper bound: 0.0754061
time: 32.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3511526, 0.3511688
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6344972, 0.6344919
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5367515, 0.5367730
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8551098, 0.8551035
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621925, 0.2621921
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9053952, 0.9053900
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3116852, 0.3116710
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634703, 1.2634610
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330900, 0.4330868
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5013086, 0.5012940

Time for backsubstitution: 6.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3385

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761415, upper bound: 0.0754341
time: 6.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761886, upper bound: 0.0753866
time: 44.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3511694, 0.3511520
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6344877, 0.6345015
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5367721, 0.5367523
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8551033, 0.8551099
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621946, 0.2621900
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9053912, 0.9053940
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3116745, 0.3116817
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634615, 1.2634698
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330870, 0.4330899
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5012954, 0.5013071

Time for backsubstitution: 6.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3385

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0761371, upper bound: 0.0754382
time: 50.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761842, upper bound: 0.0753911
time: 121.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.3995435, 0.9446968, 0.3995435, 0.9446968, -0.3511519, 0.3511688
1: -3.6258447, -2.6023710, -3.6258447, -2.6023710, -0.6344969, 0.6344920
2: -4.3699937, -3.1283081, -4.3699937, -3.1283081, -0.5367516, 0.5367730
3: -9.9915953, -7.7263937, -9.9915953, -7.7263937, -0.8551098, 0.8551022
4: -5.1987782, -3.8170004, -5.1987782, -3.8170004, -0.2621906, 0.2621951
5: -11.4803038, -8.9219160, -11.4803038, -8.9219160, -0.9053953, 0.9053894
6: -11.6431990, -9.8468781, -11.6431990, -9.8468781, -0.3116818, 0.3116757
7: -7.2156110, -4.7259231, -7.2156110, -4.7259231, -1.2634698, 1.2634618
8: 0.1018838, 0.9321412, 0.1018838, 0.9321412, -0.4330902, 0.4330868
9: -1.1751508, -0.0911610, -1.1751508, -0.0911610, -0.5013077, 0.5012953

Time for backsubstitution: 6.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3385
type: DSZ, layer: 1, pos: 3384
type: DSZ, layer: 1, pos: 3387
type: DSZ, layer: 1, pos: 461
type: DSZ, layer: 1, pos: 460
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3401
type: DSZ, layer: 1, pos: 462
type: DSZ, layer: 1, pos: 3388
type: DSZ, layer: 1, pos: 2136
type: DSZ, layer: 1, pos: 2586
type: DSZ, layer: 1, pos: 2571
type: DSZ, layer: 1, pos: 3386
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2619
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 370
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 790
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2568
type: DSZ, layer: 1, pos: 3064
type: DSZ, layer: 1, pos: 768
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 346
type: DSZ, layer: 1, pos: 602
type: DSZ, layer: 1, pos: 604
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 620
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 625
type: DSZ, layer: 1, pos: 638
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2384
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2418
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2446
type: DSZ, layer: 1, pos: 2456
type: DSZ, layer: 1, pos: 2457
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2463
type: DSZ, layer: 1, pos: 2464
type: DSZ, layer: 1, pos: 2465
type: DSZ, layer: 1, pos: 2466
type: DSZ, layer: 1, pos: 2470
type: DSZ, layer: 1, pos: 2471
type: DSZ, layer: 1, pos: 2472
type: DSZ, layer: 1, pos: 2473
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2601
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2655
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 3096
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3545
type: DSZ, layer: 1, pos: 3547
type: DSZ, layer: 1, pos: 3561
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3564

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3385

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761466, upper bound: 0.0754277
time: 289.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0761936, upper bound: 0.0753824
time: 6.52 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 302.35 seconds
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 302.35
Output dim: 8, lower bound: -0.0761197, upper bound: 0.0754544
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 302.35
Output dim: 8, lower bound: -0.0761668, upper bound: 0.0754070
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 302.35
Output dim: 8, lower bound: -0.0761153, upper bound: 0.0754575
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 302.35
Output dim: 8, lower bound: -0.0761623, upper bound: 0.0754112
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 302.35
Output dim: 8, lower bound: -0.0761247, upper bound: 0.0754492
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 302.35
Output dim: 8, lower bound: -0.0761718, upper bound: 0.0754021
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 302.35
Output dim: 8, lower bound: -0.0761203, upper bound: 0.0754527
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 302.35
Output dim: 8, lower bound: -0.0761674, upper bound: 0.0754061
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 302.35
Output dim: 8, lower bound: -0.0761415, upper bound: 0.0754341
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 302.35
Output dim: 8, lower bound: -0.0761886, upper bound: 0.0753866
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 302.35
Output dim: 8, lower bound: -0.0761371, upper bound: 0.0754382
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 302.35
Output dim: 8, lower bound: -0.0761842, upper bound: 0.0753911
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 10, time: 302.35
Output dim: 8, lower bound: -0.0761466, upper bound: 0.0754277
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 10, time: 302.35
Output dim: 8, lower bound: -0.0761936, upper bound: 0.0753824
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 302.35
Output dim: 8, lower bound: -0.0761891, upper bound: 0.0754344

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 37.87 + 3844.34 = 3882.21 seconds
