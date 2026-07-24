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
execution time: IAR + RelationalAnalysis = 7.37 + 30.26 = 37.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0762168, upper bound: 0.0762168

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3469
type: B, layer: 1, pos: 3469
type: A, layer: 1, pos: 3467
type: B, layer: 1, pos: 3467
type: A, layer: 1, pos: 3428
type: B, layer: 1, pos: 3428
type: A, layer: 1, pos: 3443
type: B, layer: 1, pos: 3443
type: A, layer: 1, pos: 3401
type: B, layer: 1, pos: 3401
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 463
type: B, layer: 1, pos: 463
type: A, layer: 1, pos: 2446
type: B, layer: 1, pos: 2446
type: A, layer: 1, pos: 3399
type: B, layer: 1, pos: 3399
type: A, layer: 1, pos: 459
type: B, layer: 1, pos: 459
type: A, layer: 1, pos: 3383
type: B, layer: 1, pos: 3383
type: A, layer: 1, pos: 2656
type: B, layer: 1, pos: 2656
type: A, layer: 1, pos: 2418
type: B, layer: 1, pos: 2418
type: A, layer: 1, pos: 3384
type: B, layer: 1, pos: 3384
type: A, layer: 1, pos: 461
type: B, layer: 1, pos: 461
type: A, layer: 1, pos: 3458
type: B, layer: 1, pos: 3458
type: A, layer: 1, pos: 2135
type: B, layer: 1, pos: 2135
type: A, layer: 1, pos: 2641
type: B, layer: 1, pos: 2641
type: A, layer: 1, pos: 3386
type: B, layer: 1, pos: 3386
type: A, layer: 1, pos: 2670
type: B, layer: 1, pos: 2670
type: A, layer: 1, pos: 3385
type: B, layer: 1, pos: 3385
type: A, layer: 1, pos: 2392
type: B, layer: 1, pos: 2392
type: A, layer: 1, pos: 2655
type: B, layer: 1, pos: 2655
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2433
type: B, layer: 1, pos: 2433
type: A, layer: 1, pos: 2419
type: B, layer: 1, pos: 2419
type: A, layer: 1, pos: 2582
type: B, layer: 1, pos: 2582
type: A, layer: 1, pos: 2420
type: B, layer: 1, pos: 2420
type: A, layer: 1, pos: 460
type: B, layer: 1, pos: 460
type: A, layer: 1, pos: 2456
type: B, layer: 1, pos: 2456
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2432
type: B, layer: 1, pos: 2432
type: A, layer: 1, pos: 2298
type: B, layer: 1, pos: 2298
type: A, layer: 1, pos: 370
type: B, layer: 1, pos: 370
type: A, layer: 1, pos: 3388
type: B, layer: 1, pos: 3388
type: A, layer: 1, pos: 2619
type: B, layer: 1, pos: 2619
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 3064
type: B, layer: 1, pos: 3064
type: A, layer: 1, pos: 2586
type: B, layer: 1, pos: 2586
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: A, layer: 1, pos: 462
type: B, layer: 1, pos: 462
type: A, layer: 1, pos: 2165
type: B, layer: 1, pos: 2165
type: A, layer: 1, pos: 2568
type: B, layer: 1, pos: 2568
type: A, layer: 1, pos: 3562
type: B, layer: 1, pos: 3562
type: A, layer: 1, pos: 3096
type: B, layer: 1, pos: 3096
type: A, layer: 1, pos: 2571
type: B, layer: 1, pos: 2571
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 2457
type: B, layer: 1, pos: 2457
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 2652
type: B, layer: 1, pos: 2652
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 815
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 2173
type: B, layer: 1, pos: 2173
type: A, layer: 1, pos: 3545
type: B, layer: 1, pos: 3545
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 2622
type: B, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 152
type: B, layer: 1, pos: 152
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 3561
type: B, layer: 1, pos: 3561
type: B, layer: 1, pos: 2598
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 98
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2599
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2685
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2685

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3469

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0758958, upper bound: 0.0762165
time: 17.28 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762158, upper bound: 0.0762159
time: 12.51 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 29.87 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 29.87
Output dim: 8, lower bound: -0.0758958, upper bound: 0.0762165
NS_A2, status: Status.UNKNOWN, split count: 1, time: 29.87
Output dim: 8, lower bound: -0.0762158, upper bound: 0.0762159

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.3997946, 0.9444311, 0.3997496, 0.9446301, -0.3511341, 0.3507161
1: -3.6250901, -2.6036658, -3.6257815, -2.6033361, -0.6328852, 0.6334584
2: -4.3683972, -3.1305227, -4.3699446, -3.1299388, -0.5340445, 0.5349391
3: -9.9898920, -7.7291870, -9.9915352, -7.7285948, -0.8515385, 0.8526696
4: -5.1981816, -3.8182025, -5.1987576, -3.8178663, -0.2601345, 0.2607324
5: -11.4782696, -8.9249659, -11.4802523, -8.9243832, -0.9013015, 0.9026042
6: -11.6427574, -9.8470869, -11.6428499, -9.8468847, -0.3116278, 0.3114561
7: -7.2128739, -4.7300158, -7.2154822, -4.7292037, -1.2577569, 1.2595829
8: 0.1052409, 0.9299629, 0.1046114, 0.9321389, -0.4298515, 0.4282463
9: -1.1742568, -0.0920116, -1.1750150, -0.0918536, -0.5001695, 0.5006558

Time for backsubstitution: 5.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3467
type: B, layer: 1, pos: 3467
type: B, layer: 1, pos: 3428
type: A, layer: 1, pos: 3428
type: B, layer: 1, pos: 3443
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 3401
type: B, layer: 1, pos: 3401
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 463
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2446
type: A, layer: 1, pos: 2446
type: A, layer: 1, pos: 3399
type: B, layer: 1, pos: 3399
type: A, layer: 1, pos: 459
type: B, layer: 1, pos: 459
type: A, layer: 1, pos: 3383
type: B, layer: 1, pos: 3383
type: A, layer: 1, pos: 2656
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2418
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3384
type: B, layer: 1, pos: 3384
type: A, layer: 1, pos: 461
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 3458
type: A, layer: 1, pos: 3458
type: B, layer: 1, pos: 2135
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 2641
type: B, layer: 1, pos: 2641
type: A, layer: 1, pos: 3386
type: B, layer: 1, pos: 3386
type: A, layer: 1, pos: 2670
type: B, layer: 1, pos: 2670
type: A, layer: 1, pos: 3385
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 2392
type: A, layer: 1, pos: 2392
type: B, layer: 1, pos: 2655
type: A, layer: 1, pos: 2655
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2433
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 2419
type: B, layer: 1, pos: 2419
type: A, layer: 1, pos: 2582
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2420
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 460
type: B, layer: 1, pos: 460
type: B, layer: 1, pos: 2456
type: A, layer: 1, pos: 2456
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2432
type: A, layer: 1, pos: 2432
type: B, layer: 1, pos: 2298
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 370
type: B, layer: 1, pos: 370
type: A, layer: 1, pos: 3388
type: B, layer: 1, pos: 3388
type: B, layer: 1, pos: 2619
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: A, layer: 1, pos: 3064
type: B, layer: 1, pos: 2586
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: A, layer: 1, pos: 462
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 2165
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2568
type: B, layer: 1, pos: 2568
type: A, layer: 1, pos: 3562
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 3096
type: A, layer: 1, pos: 3096
type: A, layer: 1, pos: 2571
type: B, layer: 1, pos: 2571
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 2652
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2173
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3545
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 2622
type: B, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 152
type: A, layer: 1, pos: 152
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 3561
type: B, layer: 1, pos: 3561
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 2598
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 98
type: A, layer: 1, pos: 98
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2599
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2685
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2685

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3467

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0756651, upper bound: 0.0762138
time: 11.78 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0758935, upper bound: 0.0762138
time: 74.83 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.3995641, 0.9446963, 0.3995601, 0.9446963, -0.3512641, 0.3516768
1: -3.6258445, -2.6024671, -3.6258442, -2.6024485, -0.6346902, 0.6343992
2: -4.3699899, -3.1283185, -4.3699903, -3.1283169, -0.5370567, 0.5355430
3: -9.9915895, -7.7265139, -9.9915915, -7.7265100, -0.8552508, 0.8529119
4: -5.1987767, -3.8170867, -5.1987777, -3.8170724, -0.2621160, 0.2618837
5: -11.4803019, -8.9220047, -11.4803019, -8.9220018, -0.9054614, 0.9022173
6: -11.6431561, -9.8468781, -11.6431646, -9.8468781, -0.3115638, 0.3120750
7: -7.2156048, -4.7259393, -7.2156057, -4.7259355, -1.2634103, 1.2588651
8: 0.1018880, 0.9321412, 0.1018871, 0.9321412, -0.4280691, 0.4330450
9: -1.1751498, -0.0911646, -1.1751502, -0.0911639, -0.5014865, 0.5010139

Time for backsubstitution: 6.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3467
type: A, layer: 1, pos: 3467
type: B, layer: 1, pos: 3428
type: A, layer: 1, pos: 3428
type: B, layer: 1, pos: 3443
type: A, layer: 1, pos: 3443
type: B, layer: 1, pos: 3401
type: A, layer: 1, pos: 3401
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2446
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: A, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: A, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: A, layer: 1, pos: 3383
type: B, layer: 1, pos: 2656
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2418
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3384
type: A, layer: 1, pos: 3384
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 461
type: A, layer: 1, pos: 461
type: B, layer: 1, pos: 3458
type: A, layer: 1, pos: 3458
type: B, layer: 1, pos: 2135
type: A, layer: 1, pos: 2135
type: B, layer: 1, pos: 2641
type: A, layer: 1, pos: 2641
type: B, layer: 1, pos: 3386
type: A, layer: 1, pos: 3386
type: B, layer: 1, pos: 2670
type: A, layer: 1, pos: 2670
type: B, layer: 1, pos: 3385
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 2392
type: B, layer: 1, pos: 2392
type: A, layer: 1, pos: 2655
type: B, layer: 1, pos: 2655
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2433
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 2419
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 2582
type: A, layer: 1, pos: 2582
type: B, layer: 1, pos: 2420
type: A, layer: 1, pos: 2420
type: B, layer: 1, pos: 460
type: A, layer: 1, pos: 460
type: A, layer: 1, pos: 2456
type: B, layer: 1, pos: 2456
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2432
type: A, layer: 1, pos: 2432
type: A, layer: 1, pos: 2298
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: A, layer: 1, pos: 370
type: B, layer: 1, pos: 3388
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 2619
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3064
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2586
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 462
type: A, layer: 1, pos: 462
type: B, layer: 1, pos: 2165
type: A, layer: 1, pos: 2165
type: B, layer: 1, pos: 2568
type: A, layer: 1, pos: 2568
type: B, layer: 1, pos: 3562
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 3096
type: B, layer: 1, pos: 3096
type: B, layer: 1, pos: 2571
type: A, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2457
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 2652
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 815
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 3545
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 2622
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 152
type: B, layer: 1, pos: 152
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 3561
type: A, layer: 1, pos: 3561
type: B, layer: 1, pos: 2598
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 98
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 2599
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2685
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2685

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3467

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762132, upper bound: 0.0759827
time: 25.40 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762131, upper bound: 0.0762133
time: 255.39 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 287.12 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 287.12
Output dim: 8, lower bound: -0.0756651, upper bound: 0.0762138
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 287.12
Output dim: 8, lower bound: -0.0758935, upper bound: 0.0762138
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 287.12
Output dim: 8, lower bound: -0.0762132, upper bound: 0.0759827
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 287.12
Output dim: 8, lower bound: -0.0762131, upper bound: 0.0762133

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: 0.3998125, 0.9444306, 0.3997638, 0.9446298, -0.3511196, 0.3507042
1: -3.6250894, -2.6039207, -3.6257811, -2.6035385, -0.6326978, 0.6332266
2: -4.3683319, -3.1305895, -4.3698940, -3.1299908, -0.5338681, 0.5347643
3: -9.9898767, -7.7301440, -9.9915218, -7.7293553, -0.8507165, 0.8518442
4: -5.1981411, -3.8183649, -5.1987257, -3.8180127, -0.2598663, 0.2604071
5: -11.4782457, -8.9263163, -11.4802370, -8.9254608, -0.9003726, 0.9015154
6: -11.6427526, -9.8475332, -11.6428461, -9.8472519, -0.3112201, 0.3109791
7: -7.2126365, -4.7321639, -7.2152996, -4.7308846, -1.2559330, 1.2573566
8: 0.1064484, 0.9299617, 0.1055550, 0.9321381, -0.4286113, 0.4272598
9: -1.1742556, -0.0926291, -1.1750141, -0.0923329, -0.4997519, 0.5001397

Time for backsubstitution: 5.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3428
type: B, layer: 1, pos: 3428
type: A, layer: 1, pos: 3443
type: B, layer: 1, pos: 3443
type: A, layer: 1, pos: 3401
type: B, layer: 1, pos: 3401
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 463
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2446
type: A, layer: 1, pos: 2446
type: A, layer: 1, pos: 3399
type: B, layer: 1, pos: 3399
type: A, layer: 1, pos: 459
type: B, layer: 1, pos: 459
type: A, layer: 1, pos: 3383
type: B, layer: 1, pos: 3383
type: A, layer: 1, pos: 2656
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2418
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3384
type: B, layer: 1, pos: 3384
type: A, layer: 1, pos: 461
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3469
type: A, layer: 1, pos: 3458
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 3467
type: B, layer: 1, pos: 2135
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 2641
type: B, layer: 1, pos: 2641
type: A, layer: 1, pos: 3386
type: B, layer: 1, pos: 3386
type: A, layer: 1, pos: 2670
type: B, layer: 1, pos: 2670
type: A, layer: 1, pos: 3385
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 2392
type: A, layer: 1, pos: 2392
type: B, layer: 1, pos: 2655
type: A, layer: 1, pos: 2655
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2433
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 2419
type: B, layer: 1, pos: 2419
type: A, layer: 1, pos: 2582
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2420
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 460
type: B, layer: 1, pos: 460
type: B, layer: 1, pos: 2456
type: A, layer: 1, pos: 2456
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2432
type: A, layer: 1, pos: 2432
type: B, layer: 1, pos: 2298
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 370
type: B, layer: 1, pos: 370
type: A, layer: 1, pos: 3388
type: B, layer: 1, pos: 3388
type: A, layer: 1, pos: 2619
type: B, layer: 1, pos: 2619
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: A, layer: 1, pos: 3064
type: B, layer: 1, pos: 2586
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: A, layer: 1, pos: 462
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 2165
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2568
type: B, layer: 1, pos: 2568
type: A, layer: 1, pos: 3562
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 3096
type: A, layer: 1, pos: 3096
type: A, layer: 1, pos: 2571
type: B, layer: 1, pos: 2571
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 2652
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2173
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3545
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 2622
type: B, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 152
type: A, layer: 1, pos: 152
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 3561
type: B, layer: 1, pos: 3561
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 2598
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 98
type: A, layer: 1, pos: 98
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2599
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2685
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2685

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3428

## Relational analysis of NS_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0754015, upper bound: 0.0762130
time: 154.10 seconds

## Relational analysis of NS_A1_A1_A2

### Relational analysis result of NS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0756632, upper bound: 0.0762130
time: 8.12 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: 0.3997886, 0.9444443, 0.3997531, 0.9446299, -0.3511337, 0.3507194
1: -3.6256826, -2.6034932, -3.6257813, -2.6033733, -0.6333092, 0.6335051
2: -4.3690729, -3.1296093, -4.3699427, -3.1299398, -0.5347471, 0.5365349
3: -9.9921618, -7.7291284, -9.9915323, -7.7286105, -0.8526511, 0.8529186
4: -5.1985426, -3.8184385, -5.1987572, -3.8180532, -0.2608213, 0.2604861
5: -11.4817429, -8.9249954, -11.4802523, -8.9244061, -0.9044418, 0.9017637
6: -11.6438951, -9.8473387, -11.6428499, -9.8471270, -0.3130645, 0.3110472
7: -7.2177715, -4.7300382, -7.2154818, -4.7292223, -1.2626057, 1.2582556
8: 0.1053112, 0.9329157, 0.1046877, 0.9321389, -0.4288488, 0.4312998
9: -1.1757072, -0.0920163, -1.1750143, -0.0918587, -0.5012177, 0.5002421

Time for backsubstitution: 5.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3428
type: B, layer: 1, pos: 3428
type: A, layer: 1, pos: 3443
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 3401
type: A, layer: 1, pos: 3401
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2446
type: B, layer: 1, pos: 2446
type: A, layer: 1, pos: 3399
type: B, layer: 1, pos: 3399
type: A, layer: 1, pos: 459
type: B, layer: 1, pos: 459
type: A, layer: 1, pos: 3383
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 2656
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2418
type: B, layer: 1, pos: 2418
type: A, layer: 1, pos: 3384
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 461
type: A, layer: 1, pos: 461
type: B, layer: 1, pos: 3469
type: A, layer: 1, pos: 3458
type: B, layer: 1, pos: 3458
type: A, layer: 1, pos: 2135
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 2641
type: A, layer: 1, pos: 2641
type: B, layer: 1, pos: 3467
type: B, layer: 1, pos: 3386
type: A, layer: 1, pos: 3386
type: B, layer: 1, pos: 2670
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 3385
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 2392
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 2655
type: B, layer: 1, pos: 2655
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2433
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 2419
type: A, layer: 1, pos: 2419
type: B, layer: 1, pos: 2582
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2420
type: B, layer: 1, pos: 2420
type: A, layer: 1, pos: 460
type: B, layer: 1, pos: 460
type: A, layer: 1, pos: 2456
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2432
type: B, layer: 1, pos: 2432
type: A, layer: 1, pos: 2298
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: A, layer: 1, pos: 370
type: B, layer: 1, pos: 3388
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 2619
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3064
type: B, layer: 1, pos: 3064
type: A, layer: 1, pos: 2586
type: B, layer: 1, pos: 2586
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 462
type: A, layer: 1, pos: 462
type: B, layer: 1, pos: 2165
type: A, layer: 1, pos: 2165
type: B, layer: 1, pos: 2568
type: A, layer: 1, pos: 2568
type: B, layer: 1, pos: 3562
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 3096
type: B, layer: 1, pos: 3096
type: A, layer: 1, pos: 2571
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2457
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 2652
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 815
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 3545
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 2622
type: A, layer: 1, pos: 2622
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 152
type: B, layer: 1, pos: 152
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 3561
type: A, layer: 1, pos: 3561
type: B, layer: 1, pos: 2598
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 98
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2599
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2685
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2685

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3428

## Relational analysis of NS_A1_A2_A1

### Relational analysis result of NS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0756312, upper bound: 0.0762132
time: 89.73 seconds

## Relational analysis of NS_A1_A2_A2

### Relational analysis result of NS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0758930, upper bound: 0.0762127
time: 76.81 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.3995782, 0.9446958, 0.3995777, 0.9446959, -0.3512523, 0.3516622
1: -3.6258433, -2.6026700, -3.6258435, -2.6027033, -0.6344582, 0.6342121
2: -4.3699389, -3.1283708, -4.3699260, -3.1283841, -0.5368807, 0.5353662
3: -9.9915771, -7.7272744, -9.9915762, -7.7274685, -0.8544242, 0.8520890
4: -5.1987472, -3.8172333, -5.1987371, -3.8172350, -0.2617908, 0.2616155
5: -11.4802828, -8.9230824, -11.4802780, -8.9233522, -0.9043729, 0.9012889
6: -11.6431551, -9.8472452, -11.6431608, -9.8473253, -0.3110869, 0.3116672
7: -7.2154207, -4.7276211, -7.2153687, -4.7280846, -1.2611842, 1.2570418
8: 0.1028315, 0.9321404, 0.1030944, 0.9321401, -0.4270827, 0.4318047
9: -1.1751487, -0.0916439, -1.1751492, -0.0917814, -0.5009707, 0.5005968

Time for backsubstitution: 5.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3428
type: A, layer: 1, pos: 3428
type: B, layer: 1, pos: 3443
type: A, layer: 1, pos: 3443
type: B, layer: 1, pos: 3401
type: A, layer: 1, pos: 3401
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2446
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: A, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: A, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: A, layer: 1, pos: 3383
type: B, layer: 1, pos: 2656
type: A, layer: 1, pos: 2656
type: B, layer: 1, pos: 2418
type: A, layer: 1, pos: 2418
type: B, layer: 1, pos: 3384
type: A, layer: 1, pos: 3384
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 461
type: A, layer: 1, pos: 461
type: B, layer: 1, pos: 3458
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 3467
type: B, layer: 1, pos: 2135
type: A, layer: 1, pos: 2135
type: B, layer: 1, pos: 2641
type: A, layer: 1, pos: 2641
type: B, layer: 1, pos: 3386
type: A, layer: 1, pos: 3386
type: B, layer: 1, pos: 2670
type: A, layer: 1, pos: 2670
type: B, layer: 1, pos: 3385
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 2392
type: B, layer: 1, pos: 2392
type: A, layer: 1, pos: 2655
type: B, layer: 1, pos: 2655
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2433
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 2419
type: B, layer: 1, pos: 2419
type: A, layer: 1, pos: 2582
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2420
type: A, layer: 1, pos: 2420
type: B, layer: 1, pos: 460
type: A, layer: 1, pos: 460
type: A, layer: 1, pos: 2456
type: B, layer: 1, pos: 2456
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2432
type: A, layer: 1, pos: 2432
type: B, layer: 1, pos: 2298
type: A, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: A, layer: 1, pos: 370
type: B, layer: 1, pos: 3388
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 2619
type: B, layer: 1, pos: 2619
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3064
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2586
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 462
type: A, layer: 1, pos: 462
type: B, layer: 1, pos: 2165
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2568
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 3562
type: A, layer: 1, pos: 3562
type: B, layer: 1, pos: 3096
type: A, layer: 1, pos: 3096
type: B, layer: 1, pos: 2571
type: A, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2457
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 2652
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 815
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 3545
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2622
type: B, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 152
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 3561
type: A, layer: 1, pos: 3561
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 98
type: A, layer: 1, pos: 98
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 2599
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2685
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2685

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3428

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762128, upper bound: 0.0757213
time: 122.33 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762128, upper bound: 0.0759830
time: 56.18 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.3995675, 0.9446959, 0.3995540, 0.9447095, -0.3512676, 0.3516763
1: -3.6258430, -2.6025047, -3.6264362, -2.6022763, -0.6347368, 0.6348231
2: -4.3699870, -3.1283193, -4.3706675, -3.1274052, -0.5386482, 0.5362458
3: -9.9915886, -7.7265296, -9.9938602, -7.7264524, -0.8554988, 0.8540246
4: -5.1987772, -3.8172746, -5.1991386, -3.8173089, -0.2618696, 0.2625705
5: -11.4803009, -8.9220285, -11.4837770, -8.9220295, -0.9046209, 0.9053577
6: -11.6431589, -9.8471203, -11.6443014, -9.8471298, -0.3111549, 0.3135116
7: -7.2156019, -4.7259583, -7.2205038, -4.7259588, -1.2620832, 1.2637129
8: 0.1019642, 0.9321412, 0.1019574, 0.9350939, -0.4311229, 0.4320423
9: -1.1751488, -0.0911698, -1.1766002, -0.0911688, -0.5010733, 0.5020624

Time for backsubstitution: 5.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3428
type: A, layer: 1, pos: 3428
type: B, layer: 1, pos: 3443
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 3401
type: B, layer: 1, pos: 3401
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 463
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2446
type: A, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: A, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: A, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 2656
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2418
type: A, layer: 1, pos: 2418
type: B, layer: 1, pos: 3384
type: A, layer: 1, pos: 3384
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 461
type: A, layer: 1, pos: 461
type: B, layer: 1, pos: 3458
type: A, layer: 1, pos: 3458
type: B, layer: 1, pos: 2135
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 2641
type: B, layer: 1, pos: 2641
type: A, layer: 1, pos: 3467
type: A, layer: 1, pos: 3386
type: B, layer: 1, pos: 3386
type: A, layer: 1, pos: 2670
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 3385
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 2392
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 2655
type: A, layer: 1, pos: 2655
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2433
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 2419
type: B, layer: 1, pos: 2419
type: A, layer: 1, pos: 2582
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2420
type: A, layer: 1, pos: 2420
type: B, layer: 1, pos: 460
type: A, layer: 1, pos: 460
type: B, layer: 1, pos: 2456
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2432
type: A, layer: 1, pos: 2432
type: B, layer: 1, pos: 2298
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 370
type: B, layer: 1, pos: 370
type: A, layer: 1, pos: 3388
type: B, layer: 1, pos: 3388
type: B, layer: 1, pos: 2619
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: A, layer: 1, pos: 3064
type: B, layer: 1, pos: 2586
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: A, layer: 1, pos: 462
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 2165
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2568
type: B, layer: 1, pos: 2568
type: A, layer: 1, pos: 3562
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 3096
type: A, layer: 1, pos: 3096
type: B, layer: 1, pos: 2571
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 2652
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2173
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3545
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 2622
type: B, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 152
type: A, layer: 1, pos: 152
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 3561
type: B, layer: 1, pos: 3561
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 2598
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 2138
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 98
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2685
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2685

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3428

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762127, upper bound: 0.0757214
time: 81.57 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762127, upper bound: 0.0762128
time: 12.07 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 99.61 seconds
NS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 99.61
Output dim: 8, lower bound: -0.0754015, upper bound: 0.0762130
NS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 99.61
Output dim: 8, lower bound: -0.0756632, upper bound: 0.0762130
NS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 99.61
Output dim: 8, lower bound: -0.0756312, upper bound: 0.0762132
NS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 99.61
Output dim: 8, lower bound: -0.0758930, upper bound: 0.0762127
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 99.61
Output dim: 8, lower bound: -0.0762128, upper bound: 0.0757213
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 99.61
Output dim: 8, lower bound: -0.0762128, upper bound: 0.0759830
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 99.61
Output dim: 8, lower bound: -0.0762127, upper bound: 0.0757214
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 99.61
Output dim: 8, lower bound: -0.0762127, upper bound: 0.0762128

## BFS NS instance: NS_A1_A1_A1

### Backsubstitution after applying NS history:
0: 0.4014934, 0.9442616, 0.4010558, 0.9445570, -0.3494424, 0.3493374
1: -3.6247945, -2.6068335, -3.6256459, -2.6057870, -0.6303366, 0.6303257
2: -4.3676825, -3.1323724, -4.3693848, -3.1312778, -0.5316356, 0.5321512
3: -9.9896698, -7.7344704, -9.9913616, -7.7327027, -0.8469017, 0.8472325
4: -5.1980524, -3.8183780, -5.1986580, -3.8180163, -0.2597634, 0.2603016
5: -11.4781446, -8.9324236, -11.4801121, -8.9301300, -0.8951648, 0.8950858
6: -11.6416292, -9.8475485, -11.6420069, -9.8472672, -0.3099063, 0.3099111
7: -7.2126999, -4.7434912, -7.2152839, -4.7395172, -1.2464644, 1.2455013
8: 0.1114113, 0.9297253, 0.1093865, 0.9321344, -0.4235516, 0.4230896
9: -1.1739870, -0.0947427, -1.1748328, -0.0939236, -0.4978856, 0.4978625

Time for backsubstitution: 5.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3443
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 3401
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 463
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2446
type: A, layer: 1, pos: 2446
type: A, layer: 1, pos: 3399
type: B, layer: 1, pos: 3399
type: A, layer: 1, pos: 459
type: B, layer: 1, pos: 459
type: A, layer: 1, pos: 3383
type: B, layer: 1, pos: 3383
type: A, layer: 1, pos: 2656
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2418
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3384
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 461
type: A, layer: 1, pos: 461
type: B, layer: 1, pos: 3469
type: A, layer: 1, pos: 3458
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 3467
type: B, layer: 1, pos: 2135
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 2641
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 3386
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 2670
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 3385
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 2392
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 2655
type: A, layer: 1, pos: 2655
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 2433
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 2419
type: B, layer: 1, pos: 2419
type: A, layer: 1, pos: 2582
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2420
type: A, layer: 1, pos: 2420
type: B, layer: 1, pos: 460
type: A, layer: 1, pos: 460
type: B, layer: 1, pos: 2456
type: A, layer: 1, pos: 2456
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2432
type: A, layer: 1, pos: 2432
type: B, layer: 1, pos: 2298
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 370
type: B, layer: 1, pos: 370
type: A, layer: 1, pos: 3388
type: B, layer: 1, pos: 3388
type: A, layer: 1, pos: 2619
type: B, layer: 1, pos: 2619
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: A, layer: 1, pos: 3064
type: B, layer: 1, pos: 2586
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: A, layer: 1, pos: 462
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 2165
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2568
type: B, layer: 1, pos: 2568
type: A, layer: 1, pos: 3562
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 3096
type: A, layer: 1, pos: 3096
type: A, layer: 1, pos: 2571
type: B, layer: 1, pos: 2571
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 2652
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2173
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3545
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 2622
type: B, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 152
type: A, layer: 1, pos: 152
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 3561
type: B, layer: 1, pos: 3561
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 2598
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 98
type: A, layer: 1, pos: 98
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2685
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2685

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3443

## Relational analysis of NS_A1_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0752797, upper bound: 0.0762108
time: 99.37 seconds

## Relational analysis of NS_A1_A1_A1_A2

### Relational analysis result of NS_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0753860, upper bound: 0.0762113
time: 23.82 seconds

## BFS NS instance: NS_A1_A1_A2

### Backsubstitution after applying NS history:
0: 0.3998155, 0.9444304, 0.3997663, 0.9446298, -0.3511156, 0.3507032
1: -3.6250894, -2.6039240, -3.6257811, -2.6035409, -0.6326963, 0.6332204
2: -4.3683271, -3.1306379, -4.3698907, -3.1300297, -0.5338212, 0.5347123
3: -9.9898758, -7.7301517, -9.9915218, -7.7293615, -0.8507317, 0.8518366
4: -5.1981397, -3.8183649, -5.1987243, -3.8180132, -0.2598635, 0.2604156
5: -11.4782467, -8.9263277, -11.4802351, -8.9254704, -0.9003897, 0.9015033
6: -11.6427269, -9.8475332, -11.6428270, -9.8472519, -0.3112113, 0.3110074
7: -7.2126355, -4.7321882, -7.2152987, -4.7309041, -1.2559223, 1.2573389
8: 0.1064524, 0.9299617, 0.1055582, 0.9321381, -0.4286020, 0.4272559
9: -1.1742556, -0.0926594, -1.1750141, -0.0923561, -0.4997721, 0.5001318

Time for backsubstitution: 5.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3443
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 3401
type: B, layer: 1, pos: 3401
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 463
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2446
type: A, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: A, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: A, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 2656
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2418
type: A, layer: 1, pos: 2418
type: B, layer: 1, pos: 3384
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 461
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 3458
type: A, layer: 1, pos: 3458
type: B, layer: 1, pos: 3467
type: B, layer: 1, pos: 2135
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 2641
type: B, layer: 1, pos: 2641
type: A, layer: 1, pos: 3386
type: B, layer: 1, pos: 3386
type: A, layer: 1, pos: 2670
type: B, layer: 1, pos: 2670
type: A, layer: 1, pos: 3385
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 2392
type: A, layer: 1, pos: 2392
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 2655
type: A, layer: 1, pos: 2655
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2433
type: A, layer: 1, pos: 2433
type: B, layer: 1, pos: 2419
type: A, layer: 1, pos: 2419
type: A, layer: 1, pos: 2582
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2420
type: A, layer: 1, pos: 2420
type: B, layer: 1, pos: 460
type: A, layer: 1, pos: 460
type: B, layer: 1, pos: 2456
type: A, layer: 1, pos: 2456
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2432
type: A, layer: 1, pos: 2432
type: B, layer: 1, pos: 2298
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 370
type: B, layer: 1, pos: 370
type: A, layer: 1, pos: 3388
type: B, layer: 1, pos: 3388
type: A, layer: 1, pos: 2619
type: B, layer: 1, pos: 2619
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: A, layer: 1, pos: 3064
type: B, layer: 1, pos: 2586
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: A, layer: 1, pos: 462
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 2165
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2568
type: B, layer: 1, pos: 2568
type: A, layer: 1, pos: 3562
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 3096
type: A, layer: 1, pos: 3096
type: A, layer: 1, pos: 2571
type: B, layer: 1, pos: 2571
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 2652
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2173
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3545
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 2622
type: B, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 152
type: A, layer: 1, pos: 152
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 3561
type: B, layer: 1, pos: 3561
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 2598
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 98
type: A, layer: 1, pos: 98
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2599
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2685
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2685

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3443

## Relational analysis of NS_A1_A1_A2_B1

### Relational analysis result of NS_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0756611, upper bound: 0.0760590
time: 38.19 seconds

## Relational analysis of NS_A1_A1_A2_B2

### Relational analysis result of NS_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0756610, upper bound: 0.0762113
time: 67.73 seconds

## BFS NS instance: NS_A1_A2_A1

### Backsubstitution after applying NS history:
0: 0.4014700, 0.9442754, 0.4010454, 0.9445570, -0.3494565, 0.3493524
1: -3.6253874, -2.6064057, -3.6256461, -2.6056216, -0.6309477, 0.6306045
2: -4.3684239, -3.1313903, -4.3694334, -3.1312270, -0.5325143, 0.5339207
3: -9.9919538, -7.7334542, -9.9913721, -7.7319584, -0.8488338, 0.8483069
4: -5.1984539, -3.8184516, -5.1986899, -3.8180568, -0.2607180, 0.2603805
5: -11.4816427, -8.9311018, -11.4801292, -8.9290743, -0.8992334, 0.8953340
6: -11.6427727, -9.8473530, -11.6420097, -9.8471413, -0.3117507, 0.3099791
7: -7.2178354, -4.7413659, -7.2154655, -4.7378550, -1.2531369, 1.2464004
8: 0.1102743, 0.9326789, 0.1085190, 0.9321353, -0.4237893, 0.4271297
9: -1.1754384, -0.0941302, -1.1748332, -0.0934494, -0.4993516, 0.4979648

Time for backsubstitution: 5.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3443
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 3401
type: A, layer: 1, pos: 3401
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2446
type: B, layer: 1, pos: 2446
type: A, layer: 1, pos: 3399
type: B, layer: 1, pos: 3399
type: A, layer: 1, pos: 459
type: B, layer: 1, pos: 459
type: A, layer: 1, pos: 3383
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 2656
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2418
type: B, layer: 1, pos: 2418
type: A, layer: 1, pos: 3384
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 461
type: A, layer: 1, pos: 461
type: B, layer: 1, pos: 3469
type: A, layer: 1, pos: 3458
type: B, layer: 1, pos: 3458
type: A, layer: 1, pos: 2135
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 2641
type: A, layer: 1, pos: 2641
type: B, layer: 1, pos: 3467
type: B, layer: 1, pos: 3386
type: A, layer: 1, pos: 3386
type: B, layer: 1, pos: 2670
type: A, layer: 1, pos: 2670
type: B, layer: 1, pos: 3385
type: A, layer: 1, pos: 3385
type: B, layer: 1, pos: 2392
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 2655
type: B, layer: 1, pos: 2655
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 3428
type: A, layer: 1, pos: 2433
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 2419
type: A, layer: 1, pos: 2419
type: B, layer: 1, pos: 2582
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2420
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 460
type: A, layer: 1, pos: 460
type: A, layer: 1, pos: 2456
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2432
type: B, layer: 1, pos: 2432
type: A, layer: 1, pos: 2298
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: A, layer: 1, pos: 370
type: B, layer: 1, pos: 3388
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 2619
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3064
type: B, layer: 1, pos: 3064
type: A, layer: 1, pos: 2586
type: B, layer: 1, pos: 2586
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 462
type: A, layer: 1, pos: 462
type: B, layer: 1, pos: 2165
type: A, layer: 1, pos: 2165
type: B, layer: 1, pos: 2568
type: A, layer: 1, pos: 2568
type: B, layer: 1, pos: 3562
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 3096
type: B, layer: 1, pos: 3096
type: A, layer: 1, pos: 2571
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2457
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 2652
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 815
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 3545
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 2622
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 152
type: B, layer: 1, pos: 152
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 3561
type: A, layer: 1, pos: 3561
type: B, layer: 1, pos: 2598
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 98
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2599
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2685
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2685

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3443

## Relational analysis of NS_A1_A2_A1_A1

### Relational analysis result of NS_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0755096, upper bound: 0.0762111
time: 98.89 seconds

## Relational analysis of NS_A1_A2_A1_A2

### Relational analysis result of NS_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0756160, upper bound: 0.0762111
time: 5.30 seconds

## BFS NS instance: NS_A1_A2_A2

### Backsubstitution after applying NS history:
0: 0.3997917, 0.9444442, 0.3997556, 0.9446298, -0.3511297, 0.3507184
1: -3.6256819, -2.6034961, -3.6257811, -2.6033759, -0.6333074, 0.6334989
2: -4.3690691, -3.1296568, -4.3699384, -3.1299784, -0.5347001, 0.5364835
3: -9.9921608, -7.7291355, -9.9915323, -7.7286158, -0.8526665, 0.8529111
4: -5.1985397, -3.8184392, -5.1987553, -3.8180532, -0.2608184, 0.2604946
5: -11.4817438, -8.9250059, -11.4802513, -8.9244156, -0.9044588, 0.9017515
6: -11.6438675, -9.8473387, -11.6428299, -9.8471270, -0.3130557, 0.3110755
7: -7.2177706, -4.7300630, -7.2154808, -4.7292409, -1.2625949, 1.2582378
8: 0.1053154, 0.9329157, 0.1046909, 0.9321388, -0.4288396, 0.4312960
9: -1.1757071, -0.0920466, -1.1750143, -0.0918819, -0.5012380, 0.5002341

Time for backsubstitution: 6.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3443
type: A, layer: 1, pos: 3443
type: B, layer: 1, pos: 3401
type: A, layer: 1, pos: 3401
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2446
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: A, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: A, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: A, layer: 1, pos: 3383
type: B, layer: 1, pos: 2656
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2418
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3384
type: A, layer: 1, pos: 3384
type: B, layer: 1, pos: 461
type: A, layer: 1, pos: 461
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 3458
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 2135
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 2641
type: A, layer: 1, pos: 2641
type: B, layer: 1, pos: 3467
type: B, layer: 1, pos: 3386
type: A, layer: 1, pos: 3386
type: B, layer: 1, pos: 2670
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 3385
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 2392
type: A, layer: 1, pos: 2392
type: B, layer: 1, pos: 3428
type: A, layer: 1, pos: 2655
type: B, layer: 1, pos: 2655
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2433
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 2419
type: A, layer: 1, pos: 2419
type: B, layer: 1, pos: 2582
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2420
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 460
type: A, layer: 1, pos: 460
type: A, layer: 1, pos: 2456
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2432
type: B, layer: 1, pos: 2432
type: A, layer: 1, pos: 2298
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: A, layer: 1, pos: 370
type: B, layer: 1, pos: 3388
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 2619
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3064
type: B, layer: 1, pos: 3064
type: A, layer: 1, pos: 2586
type: B, layer: 1, pos: 2586
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 462
type: A, layer: 1, pos: 462
type: B, layer: 1, pos: 2165
type: A, layer: 1, pos: 2165
type: B, layer: 1, pos: 2568
type: A, layer: 1, pos: 2568
type: B, layer: 1, pos: 3562
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 3096
type: B, layer: 1, pos: 3096
type: A, layer: 1, pos: 2571
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2457
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 2652
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 815
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 3545
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 2622
type: A, layer: 1, pos: 2622
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 152
type: B, layer: 1, pos: 152
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 3561
type: A, layer: 1, pos: 3561
type: B, layer: 1, pos: 2598
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 98
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2599
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2685
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2685

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3443

## Relational analysis of NS_A1_A2_A2_B1

### Relational analysis result of NS_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0758908, upper bound: 0.0760589
time: 15.87 seconds

## Relational analysis of NS_A1_A2_A2_B2

### Relational analysis result of NS_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0758908, upper bound: 0.0762117
time: 5.97 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.4008704, 0.9446233, 0.4012589, 0.9445270, -0.3498852, 0.3499849
1: -3.6257086, -2.6049180, -3.6255493, -2.6056154, -0.6315581, 0.6318508
2: -4.3694296, -3.1296570, -4.3692756, -3.1301653, -0.5342736, 0.5331365
3: -9.9914188, -7.7306223, -9.9913683, -7.7317958, -0.8498128, 0.8482745
4: -5.1986790, -3.8172383, -5.1986485, -3.8172476, -0.2616852, 0.2615125
5: -11.4801617, -8.9277515, -11.4801788, -8.9294605, -0.8979434, 0.8960810
6: -11.6423120, -9.8472605, -11.6420383, -9.8473396, -0.3100190, 0.3103535
7: -7.2154050, -4.7362533, -7.2154331, -4.7394114, -1.2493286, 1.2475729
8: 0.1066630, 0.9321367, 0.1080577, 0.9319037, -0.4229124, 0.4267451
9: -1.1749679, -0.0932347, -1.1748812, -0.0938954, -0.4986936, 0.4987303

Time for backsubstitution: 6.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3443
type: A, layer: 1, pos: 3443
type: B, layer: 1, pos: 3401
type: A, layer: 1, pos: 3401
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2446
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: A, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: A, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: A, layer: 1, pos: 3383
type: B, layer: 1, pos: 2656
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2418
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3384
type: A, layer: 1, pos: 3384
type: B, layer: 1, pos: 3469
type: A, layer: 1, pos: 461
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3458
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 3467
type: A, layer: 1, pos: 2135
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 2641
type: A, layer: 1, pos: 2641
type: B, layer: 1, pos: 3386
type: A, layer: 1, pos: 3386
type: B, layer: 1, pos: 2670
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 3385
type: B, layer: 1, pos: 3385
type: A, layer: 1, pos: 2392
type: B, layer: 1, pos: 2392
type: A, layer: 1, pos: 2655
type: B, layer: 1, pos: 2655
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 3428
type: B, layer: 1, pos: 2433
type: A, layer: 1, pos: 2433
type: B, layer: 1, pos: 2419
type: A, layer: 1, pos: 2419
type: B, layer: 1, pos: 2582
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2420
type: B, layer: 1, pos: 2420
type: A, layer: 1, pos: 460
type: B, layer: 1, pos: 460
type: A, layer: 1, pos: 2456
type: B, layer: 1, pos: 2456
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2432
type: A, layer: 1, pos: 2432
type: A, layer: 1, pos: 2298
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: A, layer: 1, pos: 370
type: B, layer: 1, pos: 3388
type: A, layer: 1, pos: 3388
type: B, layer: 1, pos: 2619
type: A, layer: 1, pos: 2619
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3064
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2586
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 462
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 2165
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2568
type: A, layer: 1, pos: 2568
type: B, layer: 1, pos: 3562
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 3096
type: B, layer: 1, pos: 3096
type: B, layer: 1, pos: 2571
type: A, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2457
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 2652
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 815
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 3545
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 2622
type: A, layer: 1, pos: 2622
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 152
type: B, layer: 1, pos: 152
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 3561
type: A, layer: 1, pos: 3561
type: B, layer: 1, pos: 2598
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 98
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2685
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2685

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3443

## Relational analysis of NS_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762106, upper bound: 0.0756003
time: 8.71 seconds

## Relational analysis of NS_A2_B1_B1_B2

### Relational analysis result of NS_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0758925, upper bound: 0.0757064
time: 8.76 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.3995806, 0.9446958, 0.3995808, 0.9446957, -0.3512513, 0.3516582
1: -3.6258433, -2.6026723, -3.6258430, -2.6027064, -0.6344525, 0.6342102
2: -4.3699360, -3.1284094, -4.3699217, -3.1284323, -0.5368288, 0.5353191
3: -9.9915771, -7.7272811, -9.9915752, -7.7274761, -0.8544168, 0.8521045
4: -5.1987453, -3.8172331, -5.1987343, -3.8172345, -0.2617993, 0.2616127
5: -11.4802828, -8.9230909, -11.4802771, -8.9233656, -0.9043608, 0.9013058
6: -11.6431351, -9.8472452, -11.6431341, -9.8473253, -0.3111152, 0.3116584
7: -7.2154202, -4.7276392, -7.2153683, -4.7281094, -1.2611662, 1.2570310
8: 0.1028347, 0.9321404, 0.1030986, 0.9321400, -0.4270788, 0.4317954
9: -1.1751488, -0.0916670, -1.1751486, -0.0918115, -0.5009626, 0.5006169

Time for backsubstitution: 6.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3443
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 3401
type: A, layer: 1, pos: 3401
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2446
type: B, layer: 1, pos: 2446
type: A, layer: 1, pos: 3399
type: B, layer: 1, pos: 3399
type: A, layer: 1, pos: 459
type: B, layer: 1, pos: 459
type: A, layer: 1, pos: 3383
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 2656
type: A, layer: 1, pos: 2656
type: B, layer: 1, pos: 2418
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3384
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 461
type: A, layer: 1, pos: 461
type: A, layer: 1, pos: 3458
type: B, layer: 1, pos: 3458
type: A, layer: 1, pos: 3467
type: B, layer: 1, pos: 2135
type: A, layer: 1, pos: 2135
type: B, layer: 1, pos: 2641
type: A, layer: 1, pos: 2641
type: B, layer: 1, pos: 3386
type: A, layer: 1, pos: 3386
type: B, layer: 1, pos: 2670
type: A, layer: 1, pos: 2670
type: B, layer: 1, pos: 3385
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 2392
type: B, layer: 1, pos: 2392
type: A, layer: 1, pos: 3428
type: A, layer: 1, pos: 2655
type: B, layer: 1, pos: 2655
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2433
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 2419
type: B, layer: 1, pos: 2419
type: A, layer: 1, pos: 2582
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2420
type: A, layer: 1, pos: 2420
type: B, layer: 1, pos: 460
type: A, layer: 1, pos: 460
type: A, layer: 1, pos: 2456
type: B, layer: 1, pos: 2456
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2432
type: A, layer: 1, pos: 2432
type: B, layer: 1, pos: 2298
type: A, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: A, layer: 1, pos: 370
type: B, layer: 1, pos: 3388
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 2619
type: B, layer: 1, pos: 2619
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3064
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2586
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 462
type: A, layer: 1, pos: 462
type: B, layer: 1, pos: 2165
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2568
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 3562
type: A, layer: 1, pos: 3562
type: B, layer: 1, pos: 3096
type: A, layer: 1, pos: 3096
type: B, layer: 1, pos: 2571
type: A, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2457
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 2652
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 815
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 3545
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 2622
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 152
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 3561
type: A, layer: 1, pos: 3561
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 2138
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 98
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 2599
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2685
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2685

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3443

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0760589, upper bound: 0.0759813
time: 5.45 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762106, upper bound: 0.0759810
time: 14.14 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.4008598, 0.9446234, 0.4012353, 0.9445407, -0.3499004, 0.3499990
1: -3.6257088, -2.6047525, -3.6261425, -2.6051884, -0.6318361, 0.6324621
2: -4.3694787, -3.1296058, -4.3700185, -3.1291842, -0.5360392, 0.5340159
3: -9.9914293, -7.7298789, -9.9936543, -7.7307796, -0.8508872, 0.8502076
4: -5.1987100, -3.8172803, -5.1990495, -3.8173220, -0.2617641, 0.2624671
5: -11.4801769, -8.9266977, -11.4836760, -8.9281378, -0.8981913, 0.9001493
6: -11.6423159, -9.8471346, -11.6431799, -9.8471441, -0.3100869, 0.3121977
7: -7.2155867, -4.7345905, -7.2205677, -4.7372861, -1.2502277, 1.2542441
8: 0.1057956, 0.9321374, 0.1069205, 0.9348572, -0.4269526, 0.4269827
9: -1.1749680, -0.0927606, -1.1763325, -0.0932826, -0.4987959, 0.5001960

Time for backsubstitution: 6.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3443
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 3401
type: B, layer: 1, pos: 3401
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 463
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2446
type: A, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: A, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: A, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 2656
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2418
type: A, layer: 1, pos: 2418
type: B, layer: 1, pos: 3384
type: A, layer: 1, pos: 3384
type: B, layer: 1, pos: 3469
type: A, layer: 1, pos: 461
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3458
type: A, layer: 1, pos: 3458
type: B, layer: 1, pos: 2135
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 2641
type: B, layer: 1, pos: 2641
type: A, layer: 1, pos: 3467
type: A, layer: 1, pos: 3386
type: B, layer: 1, pos: 3386
type: A, layer: 1, pos: 2670
type: B, layer: 1, pos: 2670
type: A, layer: 1, pos: 3385
type: B, layer: 1, pos: 3385
type: A, layer: 1, pos: 2392
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 2655
type: A, layer: 1, pos: 2655
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 3428
type: B, layer: 1, pos: 2433
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 2419
type: B, layer: 1, pos: 2419
type: A, layer: 1, pos: 2582
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2420
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 460
type: B, layer: 1, pos: 460
type: B, layer: 1, pos: 2456
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2432
type: A, layer: 1, pos: 2432
type: B, layer: 1, pos: 2298
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 370
type: B, layer: 1, pos: 370
type: A, layer: 1, pos: 3388
type: B, layer: 1, pos: 3388
type: B, layer: 1, pos: 2619
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: A, layer: 1, pos: 3064
type: B, layer: 1, pos: 2586
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: A, layer: 1, pos: 462
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 2165
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2568
type: B, layer: 1, pos: 2568
type: A, layer: 1, pos: 3562
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 3096
type: A, layer: 1, pos: 3096
type: B, layer: 1, pos: 2571
type: A, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 2652
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2173
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3545
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 2622
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 152
type: A, layer: 1, pos: 152
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 3561
type: B, layer: 1, pos: 3561
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 2598
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 2138
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 98
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2685
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2685

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3443

## Relational analysis of NS_A2_B2_B1_B1

### Relational analysis result of NS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762106, upper bound: 0.0758301
time: 8.38 seconds

## Relational analysis of NS_A2_B2_B1_B2

### Relational analysis result of NS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762106, upper bound: 0.0759358
time: 12.36 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.3995700, 0.9446959, 0.3995571, 0.9447094, -0.3512664, 0.3516721
1: -3.6258433, -2.6025069, -3.6264362, -2.6022792, -0.6347307, 0.6348217
2: -4.3699846, -3.1283579, -4.3706632, -3.1274529, -0.5385968, 0.5361989
3: -9.9915886, -7.7265358, -9.9938602, -7.7264595, -0.8554913, 0.8540401
4: -5.1987758, -3.8172746, -5.1991367, -3.8173089, -0.2618781, 0.2625677
5: -11.4802990, -8.9220371, -11.4837761, -8.9220409, -0.9046087, 0.9053748
6: -11.6431370, -9.8471203, -11.6442757, -9.8471298, -0.3111832, 0.3135027
7: -7.2156019, -4.7259769, -7.2205029, -4.7259831, -1.2620652, 1.2637022
8: 0.1019674, 0.9321412, 0.1019618, 0.9350938, -0.4311190, 0.4320329
9: -1.1751488, -0.0911929, -1.1765997, -0.0911990, -0.5010652, 0.5020826

Time for backsubstitution: 6.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3443
type: B, layer: 1, pos: 3443
type: A, layer: 1, pos: 3401
type: B, layer: 1, pos: 3401
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 463
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2446
type: A, layer: 1, pos: 2446
type: A, layer: 1, pos: 3399
type: B, layer: 1, pos: 3399
type: A, layer: 1, pos: 459
type: B, layer: 1, pos: 459
type: A, layer: 1, pos: 3383
type: B, layer: 1, pos: 3383
type: A, layer: 1, pos: 2656
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2418
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3384
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 461
type: A, layer: 1, pos: 461
type: A, layer: 1, pos: 3458
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2135
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 2641
type: B, layer: 1, pos: 2641
type: A, layer: 1, pos: 3467
type: A, layer: 1, pos: 3386
type: B, layer: 1, pos: 3386
type: A, layer: 1, pos: 2670
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 3385
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 2392
type: B, layer: 1, pos: 2392
type: A, layer: 1, pos: 3428
type: B, layer: 1, pos: 2655
type: A, layer: 1, pos: 2655
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2433
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 2419
type: B, layer: 1, pos: 2419
type: A, layer: 1, pos: 2582
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2420
type: A, layer: 1, pos: 2420
type: B, layer: 1, pos: 460
type: A, layer: 1, pos: 460
type: B, layer: 1, pos: 2456
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2432
type: A, layer: 1, pos: 2432
type: B, layer: 1, pos: 2298
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 370
type: B, layer: 1, pos: 370
type: A, layer: 1, pos: 3388
type: B, layer: 1, pos: 3388
type: B, layer: 1, pos: 2619
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: A, layer: 1, pos: 3064
type: B, layer: 1, pos: 2586
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: A, layer: 1, pos: 462
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 2165
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2568
type: B, layer: 1, pos: 2568
type: A, layer: 1, pos: 3562
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 3096
type: A, layer: 1, pos: 3096
type: B, layer: 1, pos: 2571
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 2652
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2173
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3545
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 2622
type: B, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 152
type: A, layer: 1, pos: 152
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 3561
type: B, layer: 1, pos: 3561
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 2598
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 2138
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 98
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2685
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2685

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3443

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0760588, upper bound: 0.0762112
time: 65.34 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762108, upper bound: 0.0762109
time: 489.16 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 561.11 seconds
NS_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 561.11
Output dim: 8, lower bound: -0.0752797, upper bound: 0.0762108
NS_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 561.11
Output dim: 8, lower bound: -0.0753860, upper bound: 0.0762113
NS_A1_A1_A2_B1, status: Status.VERIFIED, split count: 4, time: 561.11
Output dim: 8, lower bound: -0.0756611, upper bound: 0.0760590
NS_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 561.11
Output dim: 8, lower bound: -0.0756610, upper bound: 0.0762113
NS_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 561.11
Output dim: 8, lower bound: -0.0755096, upper bound: 0.0762111
NS_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 561.11
Output dim: 8, lower bound: -0.0756160, upper bound: 0.0762111
NS_A1_A2_A2_B1, status: Status.VERIFIED, split count: 4, time: 561.11
Output dim: 8, lower bound: -0.0758908, upper bound: 0.0760589
NS_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 561.11
Output dim: 8, lower bound: -0.0758908, upper bound: 0.0762117
NS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 561.11
Output dim: 8, lower bound: -0.0762106, upper bound: 0.0756003
NS_A2_B1_B1_B2, status: Status.VERIFIED, split count: 4, time: 561.11
Output dim: 8, lower bound: -0.0758925, upper bound: 0.0757064
NS_A2_B1_B2_A1, status: Status.VERIFIED, split count: 4, time: 561.11
Output dim: 8, lower bound: -0.0760589, upper bound: 0.0759813
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 561.11
Output dim: 8, lower bound: -0.0762106, upper bound: 0.0759810
NS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 561.11
Output dim: 8, lower bound: -0.0762106, upper bound: 0.0758301
NS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 561.11
Output dim: 8, lower bound: -0.0762106, upper bound: 0.0759358
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 561.11
Output dim: 8, lower bound: -0.0760588, upper bound: 0.0762112
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 561.11
Output dim: 8, lower bound: -0.0762108, upper bound: 0.0762109

## BFS NS instance: NS_A1_A1_A1_A1

### Backsubstitution after applying NS history:
0: 0.4029337, 0.9441808, 0.4021315, 0.9444965, -0.3480183, 0.3482731
1: -3.6246314, -2.6094337, -3.6255338, -2.6077356, -0.6283920, 0.6277461
2: -4.3675494, -3.1346791, -4.3692784, -3.1329880, -0.5294898, 0.5293509
3: -9.9894838, -7.7374935, -9.9912148, -7.7349954, -0.8437665, 0.8431439
4: -5.1979752, -3.8183842, -5.1986003, -3.8180196, -0.2596718, 0.2602251
5: -11.4780159, -8.9368849, -11.4800043, -8.9336395, -0.8908231, 0.8893569
6: -11.6406898, -9.8476171, -11.6413021, -9.8473186, -0.3086737, 0.3088535
7: -7.2126894, -4.7526989, -7.2152724, -4.7464747, -1.2390571, 1.2356275
8: 0.1155370, 0.9297214, 0.1124740, 0.9321312, -0.4192451, 0.4198331
9: -1.1737891, -0.0967230, -1.1746826, -0.0953987, -0.4963028, 0.4957671

Time for backsubstitution: 6.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3401
type: B, layer: 1, pos: 3401
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 463
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2446
type: A, layer: 1, pos: 2446
type: A, layer: 1, pos: 3399
type: B, layer: 1, pos: 3399
type: A, layer: 1, pos: 459
type: B, layer: 1, pos: 459
type: A, layer: 1, pos: 3383
type: B, layer: 1, pos: 3383
type: A, layer: 1, pos: 2656
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2418
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3384
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 461
type: A, layer: 1, pos: 461
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 3467
type: B, layer: 1, pos: 2135
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 3458
type: B, layer: 1, pos: 3458
type: A, layer: 1, pos: 2641
type: B, layer: 1, pos: 2641
type: A, layer: 1, pos: 3386
type: B, layer: 1, pos: 3386
type: A, layer: 1, pos: 2670
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 3385
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 2392
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 2655
type: A, layer: 1, pos: 2655
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2433
type: A, layer: 1, pos: 2433
type: B, layer: 1, pos: 3428
type: A, layer: 1, pos: 2419
type: B, layer: 1, pos: 2419
type: A, layer: 1, pos: 2582
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2420
type: A, layer: 1, pos: 2420
type: B, layer: 1, pos: 460
type: A, layer: 1, pos: 460
type: B, layer: 1, pos: 2456
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2432
type: A, layer: 1, pos: 2432
type: B, layer: 1, pos: 2298
type: A, layer: 1, pos: 2298
type: B, layer: 1, pos: 3443
type: A, layer: 1, pos: 370
type: B, layer: 1, pos: 370
type: A, layer: 1, pos: 3388
type: B, layer: 1, pos: 3388
type: A, layer: 1, pos: 2619
type: B, layer: 1, pos: 2619
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: A, layer: 1, pos: 3064
type: B, layer: 1, pos: 2586
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: A, layer: 1, pos: 462
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 2165
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2568
type: B, layer: 1, pos: 2568
type: A, layer: 1, pos: 3562
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 3096
type: A, layer: 1, pos: 3096
type: B, layer: 1, pos: 2571
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 2652
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2173
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3545
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 2622
type: B, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 152
type: A, layer: 1, pos: 152
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 3561
type: B, layer: 1, pos: 3561
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 2598
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 2138
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 98
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2685
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2685

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 3401

## Relational analysis of NS_A1_A1_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0750660, upper bound: 0.0762106
time: 77.75 seconds

## Relational analysis of NS_A1_A1_A1_A1_A2

### Relational analysis result of NS_A1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0752798, upper bound: 0.0762095
time: 39.23 seconds

## BFS NS instance: NS_A1_A1_A1_A2

### Backsubstitution after applying NS history:
0: 0.4014949, 0.9442616, 0.4010569, 0.9445570, -0.3494383, 0.3493370
1: -3.6247945, -2.6068342, -3.6256461, -2.6057870, -0.6303360, 0.6302809
2: -4.3676820, -3.1323748, -4.3693838, -3.1312797, -0.5316334, 0.5321487
3: -9.9896688, -7.7345300, -9.9913626, -7.7327175, -0.8469008, 0.8472161
4: -5.1980524, -3.8183780, -5.1986585, -3.8180163, -0.2597473, 0.2603016
5: -11.4781446, -8.9324284, -11.4801130, -8.9301348, -0.8951600, 0.8950545
6: -11.6416149, -9.8475609, -11.6419964, -9.8472748, -0.3099049, 0.3099136
7: -7.2126994, -4.7435751, -7.2152839, -4.7395768, -1.2464050, 1.2454123
8: 0.1114187, 0.9297252, 0.1093917, 0.9321344, -0.4235439, 0.4230845
9: -1.1739873, -0.0947448, -1.1748331, -0.0939251, -0.4978853, 0.4978505

Time for backsubstitution: 6.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3401
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 463
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2446
type: A, layer: 1, pos: 2446
type: A, layer: 1, pos: 3399
type: B, layer: 1, pos: 3399
type: A, layer: 1, pos: 459
type: B, layer: 1, pos: 459
type: A, layer: 1, pos: 3383
type: B, layer: 1, pos: 3383
type: A, layer: 1, pos: 2656
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2418
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3384
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 461
type: A, layer: 1, pos: 461
type: B, layer: 1, pos: 3469
type: A, layer: 1, pos: 3458
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 3467
type: B, layer: 1, pos: 2135
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 2641
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 3386
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 2670
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 3385
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 2392
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 2655
type: A, layer: 1, pos: 2655
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 2433
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 2419
type: B, layer: 1, pos: 2419
type: A, layer: 1, pos: 2582
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 2420
type: A, layer: 1, pos: 2420
type: B, layer: 1, pos: 460
type: A, layer: 1, pos: 460
type: B, layer: 1, pos: 2456
type: A, layer: 1, pos: 2456
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2432
type: A, layer: 1, pos: 2432
type: B, layer: 1, pos: 2298
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 370
type: B, layer: 1, pos: 370
type: A, layer: 1, pos: 3388
type: B, layer: 1, pos: 3388
type: A, layer: 1, pos: 2619
type: B, layer: 1, pos: 2619
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: A, layer: 1, pos: 3064
type: B, layer: 1, pos: 2586
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: A, layer: 1, pos: 462
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 2165
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2568
type: B, layer: 1, pos: 2568
type: A, layer: 1, pos: 3562
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 3096
type: A, layer: 1, pos: 3096
type: A, layer: 1, pos: 2571
type: B, layer: 1, pos: 2571
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 2652
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2173
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3545
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 2622
type: B, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 152
type: A, layer: 1, pos: 152
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 3561
type: B, layer: 1, pos: 3561
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 2598
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 98
type: A, layer: 1, pos: 98
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2685
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2685

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3401

## Relational analysis of NS_A1_A1_A1_A2_B1

### Relational analysis result of NS_A1_A1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0753858, upper bound: 0.0759971
time: 81.25 seconds

## Relational analysis of NS_A1_A1_A1_A2_B2

### Relational analysis result of NS_A1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0753862, upper bound: 0.0762104
time: 16.71 seconds

## BFS NS instance: NS_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: 0.3998166, 0.9444304, 0.3997678, 0.9446298, -0.3511153, 0.3506992
1: -3.6250892, -2.6039243, -3.6257811, -2.6035419, -0.6326513, 0.6332198
2: -4.3683271, -3.1306396, -4.3698897, -3.1300321, -0.5338182, 0.5347100
3: -9.9898758, -7.7301545, -9.9915228, -7.7293658, -0.8507155, 0.8518358
4: -5.1981397, -3.8183649, -5.1987233, -3.8180130, -0.2598634, 0.2603995
5: -11.4782457, -8.9263325, -11.4802361, -8.9254751, -0.9003581, 0.9014982
6: -11.6427155, -9.8475428, -11.6428099, -9.8472652, -0.3112198, 0.3110059
7: -7.2126355, -4.7322521, -7.2152987, -4.7309923, -1.2558286, 1.2572746
8: 0.1064579, 0.9299617, 0.1055659, 0.9321381, -0.4285969, 0.4272480
9: -1.1742556, -0.0926610, -1.1750141, -0.0923583, -0.4997601, 0.5001314

Time for backsubstitution: 6.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3401
type: B, layer: 1, pos: 3401
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 463
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2446
type: A, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: A, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: A, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 2656
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2418
type: A, layer: 1, pos: 2418
type: B, layer: 1, pos: 3384
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 461
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 3458
type: A, layer: 1, pos: 3458
type: B, layer: 1, pos: 3467
type: B, layer: 1, pos: 2135
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 2641
type: B, layer: 1, pos: 2641
type: A, layer: 1, pos: 3386
type: B, layer: 1, pos: 3386
type: A, layer: 1, pos: 2670
type: B, layer: 1, pos: 2670
type: A, layer: 1, pos: 3385
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 2392
type: A, layer: 1, pos: 2392
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 2655
type: A, layer: 1, pos: 2655
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2433
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 2419
type: B, layer: 1, pos: 2419
type: A, layer: 1, pos: 2582
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2420
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 460
type: B, layer: 1, pos: 460
type: B, layer: 1, pos: 2456
type: A, layer: 1, pos: 2456
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2432
type: A, layer: 1, pos: 2432
type: B, layer: 1, pos: 2298
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 370
type: B, layer: 1, pos: 370
type: A, layer: 1, pos: 3388
type: B, layer: 1, pos: 3388
type: A, layer: 1, pos: 2619
type: B, layer: 1, pos: 2619
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: A, layer: 1, pos: 3064
type: B, layer: 1, pos: 2586
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: A, layer: 1, pos: 462
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 2165
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2568
type: B, layer: 1, pos: 2568
type: A, layer: 1, pos: 3562
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 3096
type: A, layer: 1, pos: 3096
type: A, layer: 1, pos: 2571
type: B, layer: 1, pos: 2571
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: B, layer: 1, pos: 2652
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 621
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2173
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 3545
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 2622
type: B, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 152
type: A, layer: 1, pos: 152
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 3561
type: B, layer: 1, pos: 3561
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 2598
type: A, layer: 1, pos: 604
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 98
type: A, layer: 1, pos: 98
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2599
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2685
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2685

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 3401

## Relational analysis of NS_A1_A1_A2_B2_A1

### Relational analysis result of NS_A1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0754470, upper bound: 0.0762104
time: 73.73 seconds

## Relational analysis of NS_A1_A1_A2_B2_A2

### Relational analysis result of NS_A1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0756609, upper bound: 0.0762104
time: 14.19 seconds

## BFS NS instance: NS_A1_A2_A1_A1

### Backsubstitution after applying NS history:
0: 0.4029102, 0.9441946, 0.4021210, 0.9444965, -0.3480322, 0.3482882
1: -3.6252246, -2.6090062, -3.6255336, -2.6075702, -0.6290030, 0.6280246
2: -4.3682904, -3.1337008, -4.3693275, -3.1329367, -0.5303684, 0.5311188
3: -9.9917688, -7.7364788, -9.9912262, -7.7342505, -0.8456961, 0.8442183
4: -5.1983757, -3.8184578, -5.1986308, -3.8180602, -0.2606262, 0.2603041
5: -11.4815140, -8.9355640, -11.4800205, -8.9325848, -0.8948911, 0.8896049
6: -11.6418314, -9.8474216, -11.6413059, -9.8471928, -0.3105178, 0.3089215
7: -7.2178254, -4.7505736, -7.2154531, -4.7448120, -1.2457296, 1.2365261
8: 0.1143998, 0.9326747, 0.1116066, 0.9321320, -0.4194825, 0.4238737
9: -1.1752405, -0.0961105, -1.1746825, -0.0949244, -0.4977685, 0.4958695

Time for backsubstitution: 6.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3401
type: A, layer: 1, pos: 3401
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2446
type: B, layer: 1, pos: 2446
type: A, layer: 1, pos: 3399
type: B, layer: 1, pos: 3399
type: A, layer: 1, pos: 459
type: B, layer: 1, pos: 459
type: A, layer: 1, pos: 3383
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 2656
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2418
type: B, layer: 1, pos: 2418
type: A, layer: 1, pos: 3384
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 461
type: A, layer: 1, pos: 461
type: B, layer: 1, pos: 3469
type: A, layer: 1, pos: 2135
type: B, layer: 1, pos: 2135
type: A, layer: 1, pos: 3458
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2641
type: A, layer: 1, pos: 2641
type: B, layer: 1, pos: 3467
type: B, layer: 1, pos: 3386
type: A, layer: 1, pos: 3386
type: B, layer: 1, pos: 2670
type: A, layer: 1, pos: 2670
type: B, layer: 1, pos: 3385
type: A, layer: 1, pos: 3385
type: B, layer: 1, pos: 2392
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 2655
type: B, layer: 1, pos: 2655
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2433
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 2419
type: A, layer: 1, pos: 2419
type: B, layer: 1, pos: 2582
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2420
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 460
type: A, layer: 1, pos: 460
type: A, layer: 1, pos: 2456
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2432
type: B, layer: 1, pos: 2432
type: A, layer: 1, pos: 2298
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 370
type: A, layer: 1, pos: 370
type: B, layer: 1, pos: 3388
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 2619
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3064
type: B, layer: 1, pos: 3064
type: A, layer: 1, pos: 2586
type: B, layer: 1, pos: 2586
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 462
type: A, layer: 1, pos: 462
type: B, layer: 1, pos: 2165
type: A, layer: 1, pos: 2165
type: B, layer: 1, pos: 2568
type: A, layer: 1, pos: 2568
type: B, layer: 1, pos: 3562
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 3096
type: B, layer: 1, pos: 3096
type: A, layer: 1, pos: 2571
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2457
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 2652
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 815
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 3545
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 2622
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 152
type: B, layer: 1, pos: 152
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 3561
type: A, layer: 1, pos: 3561
type: B, layer: 1, pos: 2598
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 98
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2599
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2685
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2685

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3401

## Relational analysis of NS_A1_A2_A1_A1_B1

### Relational analysis result of NS_A1_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0755098, upper bound: 0.0759970
time: 60.79 seconds

## Relational analysis of NS_A1_A2_A1_A1_B2

### Relational analysis result of NS_A1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0755097, upper bound: 0.0762101
time: 127.74 seconds

## BFS NS instance: NS_A1_A2_A1_A2

### Backsubstitution after applying NS history:
0: 0.4014714, 0.9442754, 0.4010463, 0.9445570, -0.3494525, 0.3493521
1: -3.6253872, -2.6064067, -3.6256461, -2.6056221, -0.6309472, 0.6305596
2: -4.3684239, -3.1313927, -4.3694329, -3.1312284, -0.5325122, 0.5339180
3: -9.9919538, -7.7335138, -9.9913731, -7.7319727, -0.8488330, 0.8482906
4: -5.1984539, -3.8184516, -5.1986895, -3.8180568, -0.2607019, 0.2603805
5: -11.4816427, -8.9311085, -11.4801292, -8.9290800, -0.8992285, 0.8953026
6: -11.6427574, -9.8473644, -11.6419983, -9.8471498, -0.3117492, 0.3099816
7: -7.2178345, -4.7414494, -7.2154651, -4.7379141, -1.2530776, 1.2463114
8: 0.1102818, 0.9326791, 0.1085242, 0.9321352, -0.4237816, 0.4271246
9: -1.1754385, -0.0941321, -1.1748329, -0.0934508, -0.4993511, 0.4979528

Time for backsubstitution: 6.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3401
type: A, layer: 1, pos: 3401
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2446
type: B, layer: 1, pos: 2446
type: A, layer: 1, pos: 3399
type: B, layer: 1, pos: 3399
type: A, layer: 1, pos: 459
type: B, layer: 1, pos: 459
type: A, layer: 1, pos: 3383
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 2656
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2418
type: B, layer: 1, pos: 2418
type: A, layer: 1, pos: 3384
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 461
type: A, layer: 1, pos: 461
type: B, layer: 1, pos: 3469
type: A, layer: 1, pos: 3458
type: B, layer: 1, pos: 3458
type: A, layer: 1, pos: 2135
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 2641
type: A, layer: 1, pos: 2641
type: B, layer: 1, pos: 3467
type: B, layer: 1, pos: 3386
type: A, layer: 1, pos: 3386
type: B, layer: 1, pos: 2670
type: A, layer: 1, pos: 2670
type: B, layer: 1, pos: 3385
type: A, layer: 1, pos: 3385
type: B, layer: 1, pos: 2392
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 2655
type: B, layer: 1, pos: 2655
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 3428
type: A, layer: 1, pos: 2433
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 2419
type: A, layer: 1, pos: 2419
type: B, layer: 1, pos: 2582
type: A, layer: 1, pos: 2582
type: B, layer: 1, pos: 3443
type: A, layer: 1, pos: 2420
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 460
type: A, layer: 1, pos: 460
type: A, layer: 1, pos: 2456
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2432
type: B, layer: 1, pos: 2432
type: A, layer: 1, pos: 2298
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: A, layer: 1, pos: 370
type: B, layer: 1, pos: 3388
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 2619
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3064
type: B, layer: 1, pos: 3064
type: A, layer: 1, pos: 2586
type: B, layer: 1, pos: 2586
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 462
type: A, layer: 1, pos: 462
type: B, layer: 1, pos: 2165
type: A, layer: 1, pos: 2165
type: B, layer: 1, pos: 2568
type: A, layer: 1, pos: 2568
type: B, layer: 1, pos: 3562
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 3096
type: B, layer: 1, pos: 3096
type: A, layer: 1, pos: 2571
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2457
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 2652
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 815
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 3545
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 2622
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 152
type: B, layer: 1, pos: 152
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 3561
type: A, layer: 1, pos: 3561
type: B, layer: 1, pos: 2598
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 98
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2599
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2685
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2685

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3401

## Relational analysis of NS_A1_A2_A1_A2_B1

### Relational analysis result of NS_A1_A2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0756157, upper bound: 0.0759968
time: 37.30 seconds

## Relational analysis of NS_A1_A2_A1_A2_B2

### Relational analysis result of NS_A1_A2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0756157, upper bound: 0.0759967
time: 131.28 seconds

## BFS NS instance: NS_A1_A2_A2_B2

### Backsubstitution after applying NS history:
0: 0.3997927, 0.9444442, 0.3997571, 0.9446298, -0.3511295, 0.3507144
1: -3.6256821, -2.6034966, -3.6257813, -2.6033764, -0.6332627, 0.6334986
2: -4.3690691, -3.1296589, -4.3699389, -3.1299810, -0.5346973, 0.5364810
3: -9.9921608, -7.7291384, -9.9915323, -7.7286205, -0.8526499, 0.8529103
4: -5.1985397, -3.8184390, -5.1987557, -3.8180532, -0.2608184, 0.2604784
5: -11.4817448, -8.9250107, -11.4802513, -8.9244204, -0.9044272, 0.9017463
6: -11.6438551, -9.8473473, -11.6428127, -9.8471403, -0.3130642, 0.3110740
7: -7.2177711, -4.7301264, -7.2154808, -4.7293296, -1.2625015, 1.2581735
8: 0.1053209, 0.9329157, 0.1046986, 0.9321388, -0.4288345, 0.4312881
9: -1.1757070, -0.0920481, -1.1750140, -0.0918841, -0.5012259, 0.5002337

Time for backsubstitution: 6.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3401
type: A, layer: 1, pos: 3401
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2446
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: A, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: A, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: A, layer: 1, pos: 3383
type: B, layer: 1, pos: 2656
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2418
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3384
type: A, layer: 1, pos: 3384
type: B, layer: 1, pos: 461
type: A, layer: 1, pos: 461
type: B, layer: 1, pos: 3469
type: A, layer: 1, pos: 3458
type: B, layer: 1, pos: 3458
type: A, layer: 1, pos: 2135
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 2641
type: A, layer: 1, pos: 2641
type: B, layer: 1, pos: 3467
type: B, layer: 1, pos: 3386
type: A, layer: 1, pos: 3386
type: B, layer: 1, pos: 2670
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 3385
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 2392
type: A, layer: 1, pos: 2392
type: B, layer: 1, pos: 3428
type: A, layer: 1, pos: 2655
type: B, layer: 1, pos: 2655
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2433
type: B, layer: 1, pos: 2433
type: A, layer: 1, pos: 3443
type: B, layer: 1, pos: 2419
type: A, layer: 1, pos: 2419
type: B, layer: 1, pos: 2582
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2420
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 460
type: A, layer: 1, pos: 460
type: A, layer: 1, pos: 2456
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2601
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2432
type: B, layer: 1, pos: 2432
type: A, layer: 1, pos: 2298
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: A, layer: 1, pos: 370
type: B, layer: 1, pos: 3388
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 2619
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3064
type: B, layer: 1, pos: 3064
type: A, layer: 1, pos: 2586
type: B, layer: 1, pos: 2586
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 462
type: A, layer: 1, pos: 462
type: B, layer: 1, pos: 2165
type: A, layer: 1, pos: 2165
type: B, layer: 1, pos: 2568
type: A, layer: 1, pos: 2568
type: B, layer: 1, pos: 3562
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 3096
type: B, layer: 1, pos: 3096
type: A, layer: 1, pos: 2571
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2457
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 2652
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 815
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 3545
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 2622
type: A, layer: 1, pos: 2622
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 152
type: B, layer: 1, pos: 152
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 3561
type: A, layer: 1, pos: 3561
type: B, layer: 1, pos: 2598
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 98
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2599
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2685
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2685

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 3401

## Relational analysis of NS_A1_A2_A2_B2_B1

### Relational analysis result of NS_A1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0758908, upper bound: 0.0759972
time: 52.20 seconds

## Relational analysis of NS_A1_A2_A2_B2_B2

### Relational analysis result of NS_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0758908, upper bound: 0.0762107
time: 71.36 seconds

## BFS NS instance: NS_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: 0.4019460, 0.9445630, 0.4026991, 0.9444463, -0.3488206, 0.3485608
1: -3.6255965, -2.6068668, -3.6253867, -2.6082163, -0.6289783, 0.6299064
2: -4.3693237, -3.1313660, -4.3691416, -3.1324701, -0.5314779, 0.5309935
3: -9.9912701, -7.7329149, -9.9911852, -7.7348189, -0.8457243, 0.8451391
4: -5.1986203, -3.8172412, -5.1985707, -3.8172538, -0.2616087, 0.2614209
5: -11.4800520, -8.9312611, -11.4800482, -8.9339218, -0.8922143, 0.8917392
6: -11.6416092, -9.8473110, -11.6410971, -9.8474092, -0.3089615, 0.3091209
7: -7.2153940, -4.7432103, -7.2154226, -4.7486191, -1.2394547, 1.2401661
8: 0.1097508, 0.9321333, 0.1121832, 0.9318994, -0.4196561, 0.4224388
9: -1.1748178, -0.0947098, -1.1746843, -0.0958756, -0.4965978, 0.4971471

Time for backsubstitution: 6.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3401
type: A, layer: 1, pos: 3401
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2446
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: A, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: A, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: A, layer: 1, pos: 3383
type: B, layer: 1, pos: 2656
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2418
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3384
type: A, layer: 1, pos: 3384
type: B, layer: 1, pos: 3469
type: A, layer: 1, pos: 461
type: B, layer: 1, pos: 461
type: A, layer: 1, pos: 3467
type: A, layer: 1, pos: 2135
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 3458
type: A, layer: 1, pos: 3458
type: B, layer: 1, pos: 2641
type: A, layer: 1, pos: 2641
type: B, layer: 1, pos: 3386
type: A, layer: 1, pos: 3386
type: B, layer: 1, pos: 2670
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 3385
type: B, layer: 1, pos: 3385
type: A, layer: 1, pos: 2392
type: B, layer: 1, pos: 2392
type: A, layer: 1, pos: 2655
type: B, layer: 1, pos: 2655
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2445
type: A, layer: 1, pos: 2433
type: B, layer: 1, pos: 2433
type: A, layer: 1, pos: 3428
type: B, layer: 1, pos: 2419
type: A, layer: 1, pos: 2419
type: B, layer: 1, pos: 2582
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2420
type: B, layer: 1, pos: 2420
type: A, layer: 1, pos: 460
type: B, layer: 1, pos: 460
type: A, layer: 1, pos: 2456
type: B, layer: 1, pos: 2456
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3564
type: B, layer: 1, pos: 3497
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2432
type: B, layer: 1, pos: 2432
type: A, layer: 1, pos: 2298
type: B, layer: 1, pos: 2298
type: A, layer: 1, pos: 3443
type: B, layer: 1, pos: 370
type: A, layer: 1, pos: 370
type: B, layer: 1, pos: 3388
type: A, layer: 1, pos: 3388
type: B, layer: 1, pos: 2619
type: A, layer: 1, pos: 2619
type: B, layer: 1, pos: 2567
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3064
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2586
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 462
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 2165
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 2568
type: A, layer: 1, pos: 2568
type: B, layer: 1, pos: 3562
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 3096
type: B, layer: 1, pos: 3096
type: A, layer: 1, pos: 2571
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2457
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 2652
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 815
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 3545
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: B, layer: 1, pos: 2622
type: A, layer: 1, pos: 2622
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 152
type: B, layer: 1, pos: 152
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 3561
type: A, layer: 1, pos: 3561
type: B, layer: 1, pos: 2598
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 98
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 2596
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2685
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2685

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 3401

## Relational analysis of NS_A2_B1_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762108, upper bound: 0.0753859
time: 354.94 seconds

## Relational analysis of NS_A2_B1_B1_B1_B2

### Relational analysis result of NS_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762108, upper bound: 0.0755993
time: 78.58 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.3995821, 0.9446957, 0.3995820, 0.9446957, -0.3512474, 0.3516578
1: -3.6258430, -2.6026731, -3.6258430, -2.6027069, -0.6344517, 0.6341653
2: -4.3699360, -3.1284118, -4.3699207, -3.1284339, -0.5368264, 0.5353165
3: -9.9915771, -7.7272849, -9.9915752, -7.7274795, -0.8544160, 0.8520883
4: -5.1987453, -3.8172331, -5.1987348, -3.8172345, -0.2617831, 0.2616127
5: -11.4802818, -8.9230976, -11.4802780, -8.9233694, -0.9043556, 0.9012743
6: -11.6431170, -9.8472586, -11.6431217, -9.8473349, -0.3111137, 0.3116668
7: -7.2154202, -4.7277284, -7.2153678, -4.7281723, -1.2611018, 1.2569370
8: 0.1028424, 0.9321404, 0.1031043, 0.9321401, -0.4270709, 0.4317904
9: -1.1751487, -0.0916693, -1.1751485, -0.0918130, -0.5009623, 0.5006049

Time for backsubstitution: 6.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3401
type: A, layer: 1, pos: 3401
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2446
type: B, layer: 1, pos: 2446
type: A, layer: 1, pos: 3399
type: B, layer: 1, pos: 3399
type: A, layer: 1, pos: 459
type: B, layer: 1, pos: 459
type: A, layer: 1, pos: 3383
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 2656
type: A, layer: 1, pos: 2656
type: B, layer: 1, pos: 2418
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 3384
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 461
type: A, layer: 1, pos: 461
type: B, layer: 1, pos: 3458
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 3467
type: B, layer: 1, pos: 2135
type: A, layer: 1, pos: 2135
type: B, layer: 1, pos: 2641
type: A, layer: 1, pos: 2641
type: B, layer: 1, pos: 3386
type: A, layer: 1, pos: 3386
type: B, layer: 1, pos: 2670
type: A, layer: 1, pos: 2670
type: B, layer: 1, pos: 3385
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 2392
type: B, layer: 1, pos: 2392
type: A, layer: 1, pos: 3428
type: A, layer: 1, pos: 2655
type: B, layer: 1, pos: 2655
type: A, layer: 1, pos: 2445
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2433
type: A, layer: 1, pos: 2433
type: B, layer: 1, pos: 3443
type: A, layer: 1, pos: 2419
type: B, layer: 1, pos: 2419
type: A, layer: 1, pos: 2582
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2420
type: A, layer: 1, pos: 2420
type: B, layer: 1, pos: 460
type: A, layer: 1, pos: 460
type: B, layer: 1, pos: 2456
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2601
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3564
type: A, layer: 1, pos: 3564
type: A, layer: 1, pos: 3497
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: B, layer: 1, pos: 2432
type: A, layer: 1, pos: 2432
type: B, layer: 1, pos: 2298
type: A, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: A, layer: 1, pos: 370
type: B, layer: 1, pos: 3388
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 2619
type: B, layer: 1, pos: 2619
type: A, layer: 1, pos: 2567
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 531
type: A, layer: 1, pos: 531
type: B, layer: 1, pos: 3547
type: A, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3064
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2586
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 462
type: A, layer: 1, pos: 462
type: B, layer: 1, pos: 2165
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 2568
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 3562
type: A, layer: 1, pos: 3562
type: B, layer: 1, pos: 3096
type: A, layer: 1, pos: 3096
type: B, layer: 1, pos: 2571
type: A, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: A, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: A, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: A, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: A, layer: 1, pos: 618
type: B, layer: 1, pos: 581
type: A, layer: 1, pos: 581
type: B, layer: 1, pos: 638
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 2652
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 621
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 815
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 3545
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 550
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2622
type: B, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 152
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 95
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 346
type: A, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: A, layer: 1, pos: 788
type: B, layer: 1, pos: 620
type: A, layer: 1, pos: 620
type: B, layer: 1, pos: 3561
type: A, layer: 1, pos: 3561
type: A, layer: 1, pos: 2598
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 604
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 162
type: A, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: A, layer: 1, pos: 790
type: B, layer: 1, pos: 98
type: A, layer: 1, pos: 98
type: B, layer: 1, pos: 768
type: A, layer: 1, pos: 768
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 2599
type: A, layer: 1, pos: 2599
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 2596
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2384
type: A, layer: 1, pos: 2444
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2463
type: A, layer: 1, pos: 2464
type: A, layer: 1, pos: 2465
type: A, layer: 1, pos: 2466
type: A, layer: 1, pos: 2470
type: A, layer: 1, pos: 2471
type: A, layer: 1, pos: 2472
type: A, layer: 1, pos: 2473
type: A, layer: 1, pos: 2474
type: A, layer: 1, pos: 2685
type: B, layer: 1, pos: 2384
type: B, layer: 1, pos: 2444
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2463
type: B, layer: 1, pos: 2464
type: B, layer: 1, pos: 2465
type: B, layer: 1, pos: 2466
type: B, layer: 1, pos: 2470
type: B, layer: 1, pos: 2471
type: B, layer: 1, pos: 2472
type: B, layer: 1, pos: 2473
type: B, layer: 1, pos: 2474
type: B, layer: 1, pos: 2685

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 3401

## Relational analysis of NS_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762106, upper bound: 0.0757671
time: 18.64 seconds

## Relational analysis of NS_A2_B1_B2_A2_B2

### Relational analysis result of NS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762106, upper bound: 0.0757665
time: 240.76 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 37.63 + 3580.99 = 3618.62 seconds
