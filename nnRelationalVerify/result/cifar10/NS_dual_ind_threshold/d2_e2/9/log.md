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
execution time: IAR + RelationalAnalysis = 8.27 + 29.92 = 38.20 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0762168, upper bound: 0.0762168

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3469
type: A, layer: 1, pos: 3467
type: A, layer: 1, pos: 3428
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 3387
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2446
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 459
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 461
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 2432
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2419
type: A, layer: 1, pos: 460
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 3564
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 370
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 3096
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 3561
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3542
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3469

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0758958, upper bound: 0.0762165
time: 16.90 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762158, upper bound: 0.0762159
time: 12.30 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 29.27 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 29.27
Output dim: 8, lower bound: -0.0758958, upper bound: 0.0762165
NS_A2, status: Status.UNKNOWN, split count: 1, time: 29.27
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

Time for backsubstitution: 6.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3467
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 2432
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 460
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 3388
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 3096
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 3561
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
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

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0758934, upper bound: 0.0759833
time: 116.68 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0758932, upper bound: 0.0762142
time: 6.44 seconds

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

Time for backsubstitution: 6.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3467
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 2432
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 460
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 3388
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 3096
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 3561
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3467

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762132, upper bound: 0.0759827
time: 24.91 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762131, upper bound: 0.0762133
time: 251.19 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 282.51 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 282.51
Output dim: 8, lower bound: -0.0758934, upper bound: 0.0759833
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 282.51
Output dim: 8, lower bound: -0.0758932, upper bound: 0.0762142
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 282.51
Output dim: 8, lower bound: -0.0762132, upper bound: 0.0759827
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 282.51
Output dim: 8, lower bound: -0.0762131, upper bound: 0.0762133

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 0.3997982, 0.9444307, 0.3997436, 0.9446434, -0.3511375, 0.3507155
1: -3.6250894, -2.6037035, -3.6263742, -2.6031630, -0.6329321, 0.6338825
2: -4.3683944, -3.1305234, -4.3706203, -3.1290264, -0.5356394, 0.5356423
3: -9.9898911, -7.7292023, -9.9938040, -7.7285366, -0.8517876, 0.8537825
4: -5.1981812, -3.8183913, -5.1991186, -3.8181005, -0.2598882, 0.2614192
5: -11.4782677, -8.9249897, -11.4837284, -8.9244118, -0.9004609, 0.9057446
6: -11.6427574, -9.8473272, -11.6439877, -9.8471375, -0.3112189, 0.3128927
7: -7.2128711, -4.7300334, -7.2203817, -4.7292266, -1.2564297, 1.2644314
8: 0.1053172, 0.9299630, 0.1046818, 0.9350916, -0.4329052, 0.4272436
9: -1.1742556, -0.0920166, -1.1764655, -0.0918583, -0.4997558, 0.5017037

Time for backsubstitution: 6.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3428
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 3387
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2446
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3467
type: A, layer: 1, pos: 459
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 461
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 2432
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2419
type: A, layer: 1, pos: 460
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 3564
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 370
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 3096
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 3561
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3542
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3428

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0756315, upper bound: 0.0762134
time: 7.25 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0758930, upper bound: 0.0762137
time: 5.06 seconds

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

Time for backsubstitution: 6.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3428
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 3387
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2446
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3467
type: A, layer: 1, pos: 459
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 461
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 2432
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2419
type: A, layer: 1, pos: 460
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 3564
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 370
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 3096
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 3561
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3542
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3428

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0759510, upper bound: 0.0759826
time: 49.42 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762127, upper bound: 0.0759833
time: 82.22 seconds

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

Time for backsubstitution: 6.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3428
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 3387
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2446
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3467
type: A, layer: 1, pos: 459
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 461
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 2432
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2419
type: A, layer: 1, pos: 460
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 3564
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 370
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 3096
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 3561
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3542
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3428

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0759509, upper bound: 0.0762133
time: 9.48 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762128, upper bound: 0.0762128
time: 52.18 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 68.20 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 68.20
Output dim: 8, lower bound: -0.0756315, upper bound: 0.0762134
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 68.20
Output dim: 8, lower bound: -0.0758930, upper bound: 0.0762137
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 68.20
Output dim: 8, lower bound: -0.0759510, upper bound: 0.0759826
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 68.20
Output dim: 8, lower bound: -0.0762127, upper bound: 0.0759833
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 68.20
Output dim: 8, lower bound: -0.0759509, upper bound: 0.0762133
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 68.20
Output dim: 8, lower bound: -0.0762128, upper bound: 0.0762128

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.4014794, 0.9442619, 0.4010357, 0.9445707, -0.3494605, 0.3493487
1: -3.6247947, -2.6066160, -3.6262388, -2.6054115, -0.6305702, 0.6309818
2: -4.3677459, -3.1323071, -4.3701115, -3.1303113, -0.5334072, 0.5330296
3: -9.9896812, -7.7335300, -9.9936438, -7.7318850, -0.8479728, 0.8491686
4: -5.1980915, -3.8184040, -5.1990509, -3.8181043, -0.2597853, 0.2613135
5: -11.4781666, -8.9310961, -11.4836073, -8.9290810, -0.8952528, 0.8993142
6: -11.6416330, -9.8473434, -11.6431475, -9.8471518, -0.3099052, 0.3118246
7: -7.2129350, -4.7413611, -7.2203670, -4.7378592, -1.2469605, 1.2525761
8: 0.1102802, 0.9297262, 0.1085132, 0.9350880, -0.4278458, 0.4230732
9: -1.1739874, -0.0941304, -1.1762841, -0.0934491, -0.4978895, 0.4994264

Time for backsubstitution: 5.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 2432
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 460
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 3388
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 3096
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 3561
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3443

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0756157, upper bound: 0.0760587
time: 116.82 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0756158, upper bound: 0.0762109
time: 8.23 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.3998013, 0.9444306, 0.3997459, 0.9446434, -0.3511336, 0.3507146
1: -3.6250894, -2.6037064, -3.6263740, -2.6031649, -0.6329300, 0.6338764
2: -4.3683906, -3.1305721, -4.3706179, -3.1290646, -0.5355929, 0.5355902
3: -9.9898911, -7.7292099, -9.9938040, -7.7285423, -0.8518028, 0.8537749
4: -5.1981788, -3.8183913, -5.1991162, -3.8181005, -0.2598854, 0.2614277
5: -11.4782677, -8.9250002, -11.4837284, -8.9244213, -0.9004776, 0.9057323
6: -11.6427317, -9.8473272, -11.6439676, -9.8471375, -0.3112100, 0.3129210
7: -7.2128716, -4.7300582, -7.2203817, -4.7292452, -1.2564187, 1.2644135
8: 0.1053212, 0.9299630, 0.1046848, 0.9350916, -0.4328959, 0.4272399
9: -1.1742556, -0.0920467, -1.1764655, -0.0918814, -0.4997760, 0.5016959

Time for backsubstitution: 5.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 2432
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 460
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 3388
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 3096
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 3561
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
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

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0756174, upper bound: 0.0760592
time: 134.27 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0756174, upper bound: 0.0762114
time: 111.54 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.3995814, 0.9446957, 0.3995801, 0.9446958, -0.3512484, 0.3516610
1: -3.6258430, -2.6026731, -3.6258433, -2.6027055, -0.6344567, 0.6342058
2: -4.3699350, -3.1284192, -4.3699226, -3.1284225, -0.5368336, 0.5353143
3: -9.9915771, -7.7272830, -9.9915752, -7.7274742, -0.8544397, 0.8520817
4: -5.1987453, -3.8172331, -5.1987357, -3.8172350, -0.2617880, 0.2616240
5: -11.4802818, -8.9230938, -11.4802771, -8.9233618, -0.9043899, 0.9012768
6: -11.6431284, -9.8472452, -11.6431408, -9.8473253, -0.3110781, 0.3116955
7: -7.2154198, -4.7276459, -7.2153683, -4.7281036, -1.2611731, 1.2570240
8: 0.1028358, 0.9321404, 0.1030976, 0.9321401, -0.4270734, 0.4318008
9: -1.1751487, -0.0916741, -1.1751487, -0.0918047, -0.5009909, 0.5005888

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 2432
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 460
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 3388
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 3096
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 3561
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
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

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762106, upper bound: 0.0758293
time: 4.63 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762105, upper bound: 0.0758288
time: 171.11 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.4012488, 0.9445271, 0.4008460, 0.9446371, -0.3495901, 0.3503094
1: -3.6255491, -2.6054170, -3.6263015, -2.6045241, -0.6323752, 0.6319230
2: -4.3693376, -3.1301010, -4.3701582, -3.1286888, -0.5364183, 0.5336365
3: -9.9913826, -7.7308569, -9.9937010, -7.7297993, -0.8516845, 0.8494108
4: -5.1986885, -3.8172882, -5.1990714, -3.8173137, -0.2617667, 0.2624647
5: -11.4801979, -8.9281359, -11.4836550, -8.9266977, -0.8994131, 0.8989275
6: -11.6420326, -9.8471346, -11.6434631, -9.8471451, -0.3098413, 0.3124436
7: -7.2156663, -4.7372847, -7.2204885, -4.7345915, -1.2526139, 1.2518579
8: 0.1069272, 0.9319047, 0.1057889, 0.9350901, -0.4260633, 0.4278720
9: -1.1748809, -0.0932835, -1.1764193, -0.0927595, -0.4992070, 0.4997848

Time for backsubstitution: 6.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 2432
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 460
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 3388
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 3096
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 3561
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3443

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0759356, upper bound: 0.0760599
time: 7.22 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0756174, upper bound: 0.0762116
time: 134.76 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.3995706, 0.9446958, 0.3995563, 0.9447095, -0.3512636, 0.3516752
1: -3.6258430, -2.6025076, -3.6264362, -2.6022782, -0.6347355, 0.6348172
2: -4.3699832, -3.1283679, -4.3706641, -3.1274431, -0.5386018, 0.5361940
3: -9.9915886, -7.7265377, -9.9938602, -7.7264576, -0.8555140, 0.8540174
4: -5.1987753, -3.8172746, -5.1991372, -3.8173084, -0.2618668, 0.2625790
5: -11.4802999, -8.9220400, -11.4837780, -8.9220390, -0.9046378, 0.9053457
6: -11.6431303, -9.8471203, -11.6442814, -9.8471298, -0.3111461, 0.3135398
7: -7.2156019, -4.7259827, -7.2205029, -4.7259774, -1.2620720, 1.2636955
8: 0.1019682, 0.9321412, 0.1019608, 0.9350938, -0.4311135, 0.4320384
9: -1.1751487, -0.0911999, -1.1765999, -0.0911919, -0.5010935, 0.5020545

Time for backsubstitution: 6.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 2432
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 460
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 3388
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 3096
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 3561
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
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

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762106, upper bound: 0.0760600
time: 6.58 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762106, upper bound: 0.0762106
time: 13.36 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 26.34 seconds
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 26.34
Output dim: 8, lower bound: -0.0756157, upper bound: 0.0760587
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.34
Output dim: 8, lower bound: -0.0756158, upper bound: 0.0762109
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 26.34
Output dim: 8, lower bound: -0.0756174, upper bound: 0.0760592
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.34
Output dim: 8, lower bound: -0.0756174, upper bound: 0.0762114
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.34
Output dim: 8, lower bound: -0.0762106, upper bound: 0.0758293
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.34
Output dim: 8, lower bound: -0.0762105, upper bound: 0.0758288
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 26.34
Output dim: 8, lower bound: -0.0759356, upper bound: 0.0760599
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 26.34
Output dim: 8, lower bound: -0.0756174, upper bound: 0.0762116
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 26.34
Output dim: 8, lower bound: -0.0762106, upper bound: 0.0760600
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 26.34
Output dim: 8, lower bound: -0.0762106, upper bound: 0.0762106

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.4014804, 0.9442619, 0.4010371, 0.9445705, -0.3494600, 0.3493447
1: -3.6247950, -2.6066165, -3.6262386, -2.6054120, -0.6305252, 0.6309813
2: -4.3677449, -3.1323087, -4.3701115, -3.1303139, -0.5334045, 0.5330274
3: -9.9896812, -7.7335768, -9.9936438, -7.7319002, -0.8479595, 0.8491676
4: -5.1980920, -3.8184040, -5.1990509, -3.8181043, -0.2597853, 0.2612973
5: -11.4781666, -8.9310999, -11.4836082, -8.9290867, -0.8952353, 0.8993093
6: -11.6416216, -9.8473511, -11.6431322, -9.8471632, -0.3099245, 0.3118232
7: -7.2129359, -4.7414212, -7.2203665, -4.7379427, -1.2468798, 1.2525152
8: 0.1102855, 0.9297262, 0.1085205, 0.9350879, -0.4278410, 0.4230657
9: -1.1739874, -0.0941320, -1.1762842, -0.0934513, -0.4978775, 0.4994262

Time for backsubstitution: 6.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 3387
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2446
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3467
type: A, layer: 1, pos: 459
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 461
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 2432
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2419
type: A, layer: 1, pos: 460
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 3564
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 370
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 3096
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 3561
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3542
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3401

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0754021, upper bound: 0.0762105
time: 27.84 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0756157, upper bound: 0.0762108
time: 10.71 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.3998024, 0.9444307, 0.3997476, 0.9446434, -0.3511333, 0.3507106
1: -3.6250892, -2.6037073, -3.6263740, -2.6031659, -0.6328853, 0.6338757
2: -4.3683901, -3.1305737, -4.3706174, -3.1290669, -0.5355903, 0.5355879
3: -9.9898911, -7.7292128, -9.9938040, -7.7285471, -0.8517865, 0.8537741
4: -5.1981792, -3.8183913, -5.1991167, -3.8181009, -0.2598853, 0.2614115
5: -11.4782677, -8.9250040, -11.4837265, -8.9244280, -0.9004461, 0.9057273
6: -11.6427193, -9.8473377, -11.6439514, -9.8471489, -0.3112185, 0.3129195
7: -7.2128706, -4.7301226, -7.2203803, -4.7293348, -1.2563249, 1.2643491
8: 0.1053268, 0.9299629, 0.1046926, 0.9350914, -0.4328910, 0.4272320
9: -1.1742555, -0.0920483, -1.1764653, -0.0918835, -0.4997638, 0.5016955

Time for backsubstitution: 6.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 3387
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2446
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3467
type: A, layer: 1, pos: 459
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 461
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 2432
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2419
type: A, layer: 1, pos: 460
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 3564
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 370
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 3096
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 3561
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3542
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3401

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0756770, upper bound: 0.0758287
time: 91.72 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0758906, upper bound: 0.0762106
time: 54.25 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.4006705, 0.9446349, 0.4010439, 0.9446123, -0.3501623, 0.3502040
1: -3.6257303, -2.6046398, -3.6256788, -2.6053398, -0.6318405, 0.6322399
2: -4.3698287, -3.1301837, -4.3697872, -3.1308310, -0.5339780, 0.5331424
3: -9.9914312, -7.7298884, -9.9913902, -7.7309780, -0.8502650, 0.8488756
4: -5.1986847, -3.8172374, -5.1986566, -3.8172417, -0.2617107, 0.2615310
5: -11.4801712, -8.9268837, -11.4801474, -8.9284029, -0.8986084, 0.8968930
6: -11.6424103, -9.8472977, -11.6421747, -9.8473930, -0.3100218, 0.3104580
7: -7.2154088, -4.7347541, -7.2153568, -4.7376666, -1.2511028, 1.2494612
8: 0.1059723, 0.9321371, 0.1072956, 0.9321356, -0.4237780, 0.4274383
9: -1.1749974, -0.0931740, -1.1749488, -0.0938487, -0.4988375, 0.4989657

Time for backsubstitution: 6.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 3387
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2446
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3467
type: A, layer: 1, pos: 459
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 461
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 2432
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2419
type: A, layer: 1, pos: 460
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 3564
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 370
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 3096
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 3561
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3542
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3401

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0759969, upper bound: 0.0758293
time: 10.29 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762107, upper bound: 0.0758300
time: 5.34 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.3995824, 0.9446958, 0.3995816, 0.9446957, -0.3512481, 0.3516572
1: -3.6258433, -2.6026735, -3.6258430, -2.6027064, -0.6344117, 0.6342053
2: -4.3699350, -3.1284208, -4.3699226, -3.1284254, -0.5368310, 0.5353119
3: -9.9915771, -7.7272859, -9.9915752, -7.7274780, -0.8544233, 0.8520809
4: -5.1987443, -3.8172331, -5.1987348, -3.8172345, -0.2617879, 0.2616078
5: -11.4802818, -8.9230986, -11.4802780, -8.9233675, -0.9043583, 0.9012715
6: -11.6431160, -9.8472548, -11.6431227, -9.8473377, -0.3110865, 0.3116940
7: -7.2154198, -4.7277088, -7.2153683, -4.7281914, -1.2610793, 1.2569599
8: 0.1028412, 0.9321404, 0.1031054, 0.9321401, -0.4270684, 0.4317929
9: -1.1751485, -0.0916756, -1.1751487, -0.0918067, -0.5009788, 0.5005883

Time for backsubstitution: 6.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 3387
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2446
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3467
type: A, layer: 1, pos: 459
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 461
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 2432
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2419
type: A, layer: 1, pos: 460
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 3564
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 370
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 3096
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 3561
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3542
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3401

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0759969, upper bound: 0.0759808
time: 92.52 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762110, upper bound: 0.0759809
time: 68.49 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.4012499, 0.9445271, 0.4008475, 0.9446370, -0.3495899, 0.3503053
1: -3.6255491, -2.6054177, -3.6263013, -2.6045246, -0.6323303, 0.6319225
2: -4.3693376, -3.1301029, -4.3701577, -3.1286914, -0.5364156, 0.5336344
3: -9.9913826, -7.7309046, -9.9937000, -7.7298160, -0.8516712, 0.8494098
4: -5.1986885, -3.8172882, -5.1990709, -3.8173137, -0.2617666, 0.2624485
5: -11.4801979, -8.9281406, -11.4836559, -8.9267054, -0.8993956, 0.8989224
6: -11.6420212, -9.8471441, -11.6434469, -9.8471565, -0.3098606, 0.3124422
7: -7.2156658, -4.7373447, -7.2204885, -4.7346749, -1.2525332, 1.2517972
8: 0.1069325, 0.9319047, 0.1057960, 0.9350902, -0.4260584, 0.4278645
9: -1.1748810, -0.0932850, -1.1764193, -0.0927616, -0.4991949, 0.4997845

Time for backsubstitution: 6.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 3387
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2446
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3467
type: A, layer: 1, pos: 459
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 461
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 2432
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2419
type: A, layer: 1, pos: 460
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 3564
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 370
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 3096
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 3561
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3542
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

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3401

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0757218, upper bound: 0.0762102
time: 99.59 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0759357, upper bound: 0.0762117
time: 6.05 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.4006598, 0.9446349, 0.4010202, 0.9446262, -0.3501774, 0.3502181
1: -3.6257303, -2.6044741, -3.6262720, -2.6049123, -0.6321183, 0.6328512
2: -4.3698764, -3.1301322, -4.3705282, -3.1298525, -0.5357416, 0.5340222
3: -9.9914417, -7.7291431, -9.9936752, -7.7299614, -0.8513396, 0.8508087
4: -5.1987157, -3.8172793, -5.1990571, -3.8173151, -0.2617896, 0.2624857
5: -11.4801865, -8.9258299, -11.4836464, -8.9270792, -0.8988562, 0.9009614
6: -11.6424122, -9.8471718, -11.6433172, -9.8471985, -0.3100899, 0.3123022
7: -7.2155905, -4.7330904, -7.2204914, -4.7355413, -1.2520016, 1.2561324
8: 0.1051049, 0.9321377, 0.1061584, 0.9350895, -0.4278183, 0.4276760
9: -1.1749973, -0.0927000, -1.1764004, -0.0932359, -0.4989400, 0.5004315

Time for backsubstitution: 6.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 3387
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2446
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3467
type: A, layer: 1, pos: 459
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 461
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 2432
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2419
type: A, layer: 1, pos: 460
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 3564
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 370
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 3096
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 3561
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3542
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3401

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0759968, upper bound: 0.0760587
time: 106.03 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762105, upper bound: 0.0758287
time: 125.55 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.3995717, 0.9446959, 0.3995578, 0.9447094, -0.3512634, 0.3516711
1: -3.6258430, -2.6025085, -3.6264362, -2.6022794, -0.6346903, 0.6348167
2: -4.3699832, -3.1283698, -4.3706636, -3.1274459, -0.5385991, 0.5361916
3: -9.9915886, -7.7265406, -9.9938602, -7.7264614, -0.8554978, 0.8540164
4: -5.1987748, -3.8172750, -5.1991363, -3.8173089, -0.2618668, 0.2625628
5: -11.4802990, -8.9220448, -11.4837761, -8.9220448, -0.9046062, 0.9053406
6: -11.6431189, -9.8471298, -11.6442652, -9.8471422, -0.3111546, 0.3135384
7: -7.2156019, -4.7260461, -7.2205029, -4.7260661, -1.2619781, 1.2636307
8: 0.1019739, 0.9321412, 0.1019683, 0.9350938, -0.4311085, 0.4320306
9: -1.1751487, -0.0912014, -1.1766000, -0.0911942, -0.5010814, 0.5020540

Time for backsubstitution: 6.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3401
type: A, layer: 1, pos: 3387
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2446
type: A, layer: 1, pos: 3399
type: A, layer: 1, pos: 3467
type: A, layer: 1, pos: 459
type: A, layer: 1, pos: 3383
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 3384
type: A, layer: 1, pos: 2418
type: A, layer: 1, pos: 461
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 3458
type: A, layer: 1, pos: 3386
type: A, layer: 1, pos: 2641
type: A, layer: 1, pos: 2135
type: A, layer: 1, pos: 3385
type: A, layer: 1, pos: 2670
type: A, layer: 1, pos: 2445
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 2655
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 2432
type: A, layer: 1, pos: 2582
type: A, layer: 1, pos: 2419
type: A, layer: 1, pos: 460
type: A, layer: 1, pos: 2420
type: A, layer: 1, pos: 2456
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 2601
type: A, layer: 1, pos: 3564
type: A, layer: 1, pos: 3388
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 370
type: A, layer: 1, pos: 2619
type: A, layer: 1, pos: 531
type: A, layer: 1, pos: 2567
type: A, layer: 1, pos: 462
type: A, layer: 1, pos: 3547
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3064
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2586
type: A, layer: 1, pos: 2568
type: A, layer: 1, pos: 3096
type: A, layer: 1, pos: 2165
type: A, layer: 1, pos: 3562
type: A, layer: 1, pos: 2571
type: A, layer: 1, pos: 625
type: A, layer: 1, pos: 639
type: A, layer: 1, pos: 2457
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 2652
type: A, layer: 1, pos: 638
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 3545
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 3561
type: A, layer: 1, pos: 152
type: A, layer: 1, pos: 620
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 346
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 604
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2136
type: A, layer: 1, pos: 162
type: A, layer: 1, pos: 790
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 768
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 2599
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 3542
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3401

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0759971, upper bound: 0.0762098
time: 86.59 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762105, upper bound: 0.0762113
time: 94.89 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 188.01 seconds
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 188.01
Output dim: 8, lower bound: -0.0754021, upper bound: 0.0762105
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 188.01
Output dim: 8, lower bound: -0.0756157, upper bound: 0.0762108
NS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 188.01
Output dim: 8, lower bound: -0.0756770, upper bound: 0.0758287
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 188.01
Output dim: 8, lower bound: -0.0758906, upper bound: 0.0762106
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 188.01
Output dim: 8, lower bound: -0.0759969, upper bound: 0.0758293
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 188.01
Output dim: 8, lower bound: -0.0762107, upper bound: 0.0758300
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 188.01
Output dim: 8, lower bound: -0.0759969, upper bound: 0.0759808
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 188.01
Output dim: 8, lower bound: -0.0762110, upper bound: 0.0759809
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 188.01
Output dim: 8, lower bound: -0.0757218, upper bound: 0.0762102
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 188.01
Output dim: 8, lower bound: -0.0759357, upper bound: 0.0762117
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 188.01
Output dim: 8, lower bound: -0.0759968, upper bound: 0.0760587
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 188.01
Output dim: 8, lower bound: -0.0762105, upper bound: 0.0758287
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 188.01
Output dim: 8, lower bound: -0.0759971, upper bound: 0.0762098
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 188.01
Output dim: 8, lower bound: -0.0762105, upper bound: 0.0762113

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.4040297, 0.9438607, 0.4028044, 0.9444627, -0.3468591, 0.3472720
1: -3.6241620, -2.6105878, -3.6260338, -2.6081548, -0.6273354, 0.6269389
2: -4.3665500, -3.1325617, -4.3692932, -3.1303878, -0.5320880, 0.5319281
3: -9.9887781, -7.7385583, -9.9934044, -7.7353511, -0.8435813, 0.8439156
4: -5.1980157, -3.8194532, -5.1990261, -3.8188274, -0.2587356, 0.2601067
5: -11.4772301, -8.9371614, -11.4834108, -8.9332962, -0.8900856, 0.8930478
6: -11.6415415, -9.8476210, -11.6430731, -9.8473473, -0.3095757, 0.3114561
7: -7.2126231, -4.7483749, -7.2203293, -4.7427902, -1.2411186, 1.2452681
8: 0.1154299, 0.9291447, 0.1120969, 0.9350818, -0.4226897, 0.4189004
9: -1.1733080, -0.0976130, -1.1760106, -0.0958727, -0.4950233, 0.4958757

Time for backsubstitution: 6.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 2432
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 460
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 3388
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 3096
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 3561
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3387

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0754018, upper bound: 0.0761080
time: 40.11 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0754019, upper bound: 0.0762107
time: 10.40 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.4014819, 0.9442618, 0.4010380, 0.9445706, -0.3494577, 0.3493445
1: -3.6247945, -2.6066186, -3.6262388, -2.6054134, -0.6305247, 0.6309569
2: -4.3677444, -3.1323085, -4.3701115, -3.1303141, -0.5334018, 0.5330273
3: -9.9896822, -7.7335787, -9.9936438, -7.7319021, -0.8479592, 0.8491620
4: -5.1980920, -3.8184161, -5.1990509, -3.8181119, -0.2597930, 0.2612963
5: -11.4781666, -8.9311028, -11.4836073, -8.9290886, -0.8952348, 0.8993043
6: -11.6416216, -9.8473549, -11.6431322, -9.8471661, -0.3099245, 0.3118175
7: -7.2129359, -4.7414503, -7.2203665, -4.7379608, -1.2468871, 1.2525138
8: 0.1102860, 0.9297262, 0.1085208, 0.9350880, -0.4278403, 0.4230664
9: -1.1739872, -0.0941339, -1.1762841, -0.0934523, -0.4978772, 0.4994162

Time for backsubstitution: 6.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 2432
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 460
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 3388
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 3096
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 3561
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3387

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0756157, upper bound: 0.0761085
time: 7.92 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0756158, upper bound: 0.0759807
time: 131.93 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.3998039, 0.9444307, 0.3997484, 0.9446434, -0.3511308, 0.3507105
1: -3.6250892, -2.6037092, -3.6263738, -2.6031671, -0.6328851, 0.6338512
2: -4.3683896, -3.1305740, -4.3706174, -3.1290674, -0.5355875, 0.5355878
3: -9.9898911, -7.7292156, -9.9938040, -7.7285490, -0.8517861, 0.8537688
4: -5.1981792, -3.8184028, -5.1991167, -3.8181081, -0.2598931, 0.2614105
5: -11.4782686, -8.9250078, -11.4837265, -8.9244289, -0.9004456, 0.9057220
6: -11.6427193, -9.8473415, -11.6439514, -9.8471527, -0.3112184, 0.3129138
7: -7.2128706, -4.7301512, -7.2203803, -4.7293520, -1.2563323, 1.2643478
8: 0.1053274, 0.9299629, 0.1046930, 0.9350915, -0.4328902, 0.4272326
9: -1.1742557, -0.0920502, -1.1764650, -0.0918848, -0.4997638, 0.5016856

Time for backsubstitution: 6.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 2432
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 460
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 3388
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 3096
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 3561
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
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
type: B, layer: 1, pos: 3387

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0758906, upper bound: 0.0761088
time: 5.66 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0758906, upper bound: 0.0762104
time: 5.58 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.4006718, 0.9446350, 0.4010447, 0.9446124, -0.3501598, 0.3502038
1: -3.6257303, -2.6046422, -3.6256788, -2.6053410, -0.6318403, 0.6322151
2: -4.3698282, -3.1301837, -4.3697872, -3.1308312, -0.5339752, 0.5331423
3: -9.9914303, -7.7298908, -9.9913902, -7.7309790, -0.8502647, 0.8488703
4: -5.1986852, -3.8172493, -5.1986570, -3.8172498, -0.2617185, 0.2615300
5: -11.4801731, -8.9268875, -11.4801474, -8.9284029, -0.8986080, 0.8968878
6: -11.6424103, -9.8473015, -11.6421747, -9.8473969, -0.3100218, 0.3104522
7: -7.2154088, -4.7347822, -7.2153568, -4.7376847, -1.2511102, 1.2494597
8: 0.1059728, 0.9321371, 0.1072960, 0.9321356, -0.4237774, 0.4274391
9: -1.1749972, -0.0931759, -1.1749487, -0.0938497, -0.4988372, 0.4989557

Time for backsubstitution: 6.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 2432
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 460
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 3388
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 3096
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 3561
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3387

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0754034, upper bound: 0.0757268
time: 9.60 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762104, upper bound: 0.0758290
time: 25.71 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.3995838, 0.9446958, 0.3995824, 0.9446957, -0.3512457, 0.3516570
1: -3.6258433, -2.6026754, -3.6258430, -2.6027076, -0.6344115, 0.6341809
2: -4.3699350, -3.1284208, -4.3699217, -3.1284254, -0.5368283, 0.5353118
3: -9.9915771, -7.7272882, -9.9915752, -7.7274809, -0.8544229, 0.8520755
4: -5.1987443, -3.8172452, -5.1987348, -3.8172417, -0.2617957, 0.2616068
5: -11.4802809, -8.9231014, -11.4802780, -8.9233694, -0.9043579, 0.9012665
6: -11.6431160, -9.8472586, -11.6431227, -9.8473396, -0.3110865, 0.3116883
7: -7.2154198, -4.7277379, -7.2153683, -4.7282100, -1.2610869, 1.2569582
8: 0.1028419, 0.9321404, 0.1031058, 0.9321400, -0.4270676, 0.4317937
9: -1.1751488, -0.0916776, -1.1751487, -0.0918078, -0.5009786, 0.5005783

Time for backsubstitution: 6.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 2432
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 460
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 3388
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 3096
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 3561
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
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
type: B, layer: 1, pos: 3387

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762105, upper bound: 0.0758786
time: 6.91 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762109, upper bound: 0.0759814
time: 6.45 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.4037990, 0.9441262, 0.4026147, 0.9445293, -0.3469886, 0.3482328
1: -3.6249170, -2.6093881, -3.6260977, -2.6072679, -0.6291412, 0.6278800
2: -4.3681417, -3.1303566, -4.3693390, -3.1287661, -0.5350986, 0.5325348
3: -9.9904804, -7.7358861, -9.9934616, -7.7332668, -0.8472933, 0.8441577
4: -5.1986122, -3.8183379, -5.1990452, -3.8180358, -0.2607169, 0.2612577
5: -11.4792604, -8.9342012, -11.4834585, -8.9309130, -0.8942461, 0.8926608
6: -11.6419411, -9.8474131, -11.6433878, -9.8473406, -0.3095118, 0.3120751
7: -7.2153544, -4.7442985, -7.2204514, -4.7395220, -1.2467716, 1.2445500
8: 0.1120769, 0.9313230, 0.1093724, 0.9350842, -0.4209074, 0.4236993
9: -1.1742036, -0.0967661, -1.1761467, -0.0951833, -0.4963407, 0.4962335

Time for backsubstitution: 6.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 2432
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 460
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 3388
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 3096
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 3561
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3387

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0757215, upper bound: 0.0761094
time: 4.88 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0757216, upper bound: 0.0762108
time: 9.88 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.4012513, 0.9445271, 0.4008484, 0.9446370, -0.3495874, 0.3503052
1: -3.6255491, -2.6054199, -3.6263013, -2.6045265, -0.6323301, 0.6318980
2: -4.3693371, -3.1301029, -4.3701577, -3.1286914, -0.5364130, 0.5336344
3: -9.9913826, -7.7309074, -9.9937010, -7.7298179, -0.8516706, 0.8494046
4: -5.1986885, -3.8172996, -5.1990714, -3.8173213, -0.2617744, 0.2624475
5: -11.4801979, -8.9281435, -11.4836559, -8.9267063, -0.8993951, 0.8989174
6: -11.6420212, -9.8471470, -11.6434469, -9.8471584, -0.3098606, 0.3124364
7: -7.2156658, -4.7373738, -7.2204885, -4.7346931, -1.2525407, 1.2517955
8: 0.1069329, 0.9319047, 0.1057964, 0.9350902, -0.4260577, 0.4278652
9: -1.1748806, -0.0932868, -1.1764195, -0.0927628, -0.4991948, 0.4997746

Time for backsubstitution: 6.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 2432
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 460
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 3388
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 3096
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 3561
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
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
type: B, layer: 1, pos: 3387

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0754034, upper bound: 0.0761087
time: 32.64 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0759356, upper bound: 0.0762105
time: 100.46 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.4006613, 0.9446350, 0.4010211, 0.9446261, -0.3501750, 0.3502179
1: -3.6257303, -2.6044762, -3.6262722, -2.6049132, -0.6321181, 0.6328264
2: -4.3698764, -3.1301324, -4.3705282, -3.1298525, -0.5357390, 0.5340221
3: -9.9914417, -7.7291470, -9.9936752, -7.7299643, -0.8513393, 0.8508033
4: -5.1987157, -3.8172920, -5.1990576, -3.8173227, -0.2617974, 0.2624846
5: -11.4801865, -8.9258327, -11.4836464, -8.9270811, -0.8988557, 0.9009562
6: -11.6424122, -9.8471756, -11.6433172, -9.8472004, -0.3100898, 0.3122965
7: -7.2155905, -4.7331195, -7.2204914, -4.7355595, -1.2520092, 1.2561309
8: 0.1051056, 0.9321377, 0.1061589, 0.9350895, -0.4278176, 0.4276767
9: -1.1749972, -0.0927017, -1.1764003, -0.0932371, -0.4989397, 0.5004215

Time for backsubstitution: 6.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 2432
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 460
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 3388
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 3096
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 3561
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3387

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762107, upper bound: 0.0759563
time: 37.07 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0762108, upper bound: 0.0758290
time: 121.81 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.4021212, 0.9442964, 0.4013252, 0.9446027, -0.3486650, 0.3496030
1: -3.6252151, -2.6064796, -3.6262338, -2.6050222, -0.6315005, 0.6307739
2: -4.3687873, -3.1286240, -4.3698444, -3.1275206, -0.5372813, 0.5350912
3: -9.9906874, -7.7315221, -9.9936218, -7.7299123, -0.8511212, 0.8487646
4: -5.1986985, -3.8183250, -5.1991110, -3.8180320, -0.2608170, 0.2613720
5: -11.4793615, -8.9281025, -11.4835806, -8.9262524, -0.8994577, 0.8990794
6: -11.6430359, -9.8473988, -11.6442060, -9.8473263, -0.3108056, 0.3131710
7: -7.2152901, -4.7330008, -7.2204666, -4.7309141, -1.2562168, 1.2563835
8: 0.1071050, 0.9315594, 0.1055356, 0.9350878, -0.4259694, 0.4278767
9: -1.1744736, -0.0946826, -1.1763294, -0.0936157, -0.4982289, 0.4985043

Time for backsubstitution: 6.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 3469
type: B, layer: 1, pos: 2446
type: B, layer: 1, pos: 3399
type: B, layer: 1, pos: 459
type: B, layer: 1, pos: 3383
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 3384
type: B, layer: 1, pos: 2418
type: B, layer: 1, pos: 461
type: B, layer: 1, pos: 3458
type: B, layer: 1, pos: 2641
type: B, layer: 1, pos: 3386
type: B, layer: 1, pos: 2135
type: B, layer: 1, pos: 3401
type: B, layer: 1, pos: 2670
type: B, layer: 1, pos: 2445
type: B, layer: 1, pos: 3385
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 2655
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 2432
type: B, layer: 1, pos: 2582
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 2420
type: B, layer: 1, pos: 2456
type: B, layer: 1, pos: 2134
type: B, layer: 1, pos: 2601
type: B, layer: 1, pos: 460
type: B, layer: 1, pos: 3564
type: B, layer: 1, pos: 3388
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 370
type: B, layer: 1, pos: 2619
type: B, layer: 1, pos: 531
type: B, layer: 1, pos: 2567
type: B, layer: 1, pos: 3547
type: B, layer: 1, pos: 462
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3064
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2586
type: B, layer: 1, pos: 2568
type: B, layer: 1, pos: 3096
type: B, layer: 1, pos: 2165
type: B, layer: 1, pos: 3562
type: B, layer: 1, pos: 2571
type: B, layer: 1, pos: 625
type: B, layer: 1, pos: 639
type: B, layer: 1, pos: 2457
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 2652
type: B, layer: 1, pos: 638
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 3545
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 95
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 3561
type: B, layer: 1, pos: 152
type: B, layer: 1, pos: 620
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 346
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 604
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2136
type: B, layer: 1, pos: 162
type: B, layer: 1, pos: 790
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 768
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 2599
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 2566
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

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3387

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0759967, upper bound: 0.0759553
time: 717.45 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0759966, upper bound: 0.0760586
time: 9.88 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 38.20 + 3675.33 = 3713.53 seconds
