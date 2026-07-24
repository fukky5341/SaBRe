## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 14)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.1342351305


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.2664312, 0.3793331, -0.2664312, 0.3793331, -0.4741454, 0.4741454)
1: (-1.4844748, -0.3354728, -1.4844748, -0.3354728, -0.4965302, 0.4965302)
2: (-2.2458954, -0.9845571, -2.2458954, -0.9845571, -0.9277935, 0.9277936)
3: (-5.1692300, -2.8711410, -5.1692300, -2.8711410, -1.2754086, 1.2754087)
4: (-2.7080197, -1.1290655, -2.7080197, -1.1290655, -0.6519085, 0.6519084)
5: (-5.5053887, -3.0804513, -5.5053887, -3.0804513, -1.3902712, 1.3902712)
6: (-5.3754797, -3.5238063, -5.3754797, -3.5238063, -0.4283794, 0.4283794)
7: (-4.6394372, -2.3089204, -4.6394372, -2.3089204, -1.4446464, 1.4446465)
8: (0.3549576, 1.0467505, 0.3549576, 1.0467505, -0.5474601, 0.5474601)
9: (-1.3568701, 0.1132612, -1.3568701, 0.1132612, -1.3206272, 1.3206272)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.79 + 293.49 = 301.28 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.1343695, upper bound: 0.1343626

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3429
type: A, layer: 1, pos: 3442
type: A, layer: 1, pos: 380
type: A, layer: 1, pos: 3444
type: A, layer: 1, pos: 3427
type: A, layer: 1, pos: 3428
type: A, layer: 1, pos: 3413
type: A, layer: 1, pos: 3412
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 3446
type: A, layer: 1, pos: 3408
type: A, layer: 1, pos: 489
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 3422
type: A, layer: 1, pos: 3445
type: A, layer: 1, pos: 3423
type: A, layer: 1, pos: 483
type: A, layer: 1, pos: 3425
type: A, layer: 1, pos: 3409
type: A, layer: 1, pos: 487
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 522
type: A, layer: 1, pos: 3414
type: A, layer: 1, pos: 3410
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 3443
type: A, layer: 1, pos: 3411
type: A, layer: 1, pos: 3424
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 485
type: A, layer: 1, pos: 521
type: A, layer: 1, pos: 523
type: A, layer: 1, pos: 506
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 3438
type: A, layer: 1, pos: 504
type: A, layer: 1, pos: 3430
type: A, layer: 1, pos: 486
type: A, layer: 1, pos: 505
type: A, layer: 1, pos: 517
type: A, layer: 1, pos: 490
type: A, layer: 1, pos: 3461
type: A, layer: 1, pos: 3462
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 3436
type: A, layer: 1, pos: 484
type: A, layer: 1, pos: 503
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 473
type: A, layer: 1, pos: 2433
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 3439
type: A, layer: 1, pos: 474
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 500
type: A, layer: 1, pos: 472
type: A, layer: 1, pos: 470
type: A, layer: 1, pos: 3460
type: A, layer: 1, pos: 469
type: A, layer: 1, pos: 2432
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2658
type: A, layer: 1, pos: 499
type: A, layer: 1, pos: 2193
type: A, layer: 1, pos: 537
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 3440
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 2226
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 513
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 3452
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 519
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 3497
type: A, layer: 1, pos: 2152
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 3451
type: A, layer: 1, pos: 2231
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 471
type: A, layer: 1, pos: 573
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 2647
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 3124
type: A, layer: 1, pos: 143
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2195
type: A, layer: 1, pos: 572
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 2151
type: A, layer: 1, pos: 3231
type: A, layer: 1, pos: 2606
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 2170
type: A, layer: 1, pos: 577
type: A, layer: 1, pos: 3496
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 3053
type: A, layer: 1, pos: 2614
type: A, layer: 1, pos: 3453
type: A, layer: 1, pos: 3055
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 2651
type: A, layer: 1, pos: 2163
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2388
type: A, layer: 1, pos: 3516
type: A, layer: 1, pos: 2555
type: A, layer: 1, pos: 2169
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 507
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 2682
type: A, layer: 1, pos: 3475
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 2569
type: A, layer: 1, pos: 510
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 3490
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 3450
type: A, layer: 1, pos: 571
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 3245
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 535
type: A, layer: 1, pos: 3465
type: A, layer: 1, pos: 849
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 3420
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 3482
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 2346
type: A, layer: 1, pos: 3503
type: A, layer: 1, pos: 3467
type: A, layer: 1, pos: 525
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 3250
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 555
type: A, layer: 1, pos: 432
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 371
type: A, layer: 1, pos: 526
type: A, layer: 1, pos: 527
type: A, layer: 1, pos: 3474
type: A, layer: 1, pos: 2140
type: A, layer: 1, pos: 556
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 528
type: A, layer: 1, pos: 558
type: A, layer: 1, pos: 541
type: A, layer: 1, pos: 3495
type: A, layer: 1, pos: 550
type: A, layer: 1, pos: 543
type: A, layer: 1, pos: 3466
type: A, layer: 1, pos: 2123
type: A, layer: 1, pos: 3481
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 3468
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 435
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 2460
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2697

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3429

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1342269, upper bound: 0.1343436
time: 195.36 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1342269, upper bound: 0.1343476
time: 433.24 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 628.66 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 628.66
Output dim: 8, lower bound: -0.1342269, upper bound: 0.1343436
NS_A2, status: Status.UNKNOWN, split count: 1, time: 628.66
Output dim: 8, lower bound: -0.1342269, upper bound: 0.1343476

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.2641892, 0.3779109, -0.2646301, 0.3793172, -0.4718606, 0.4707689
1: -1.4823892, -0.3377724, -1.4844687, -0.3374171, -0.4926649, 0.4943070
2: -2.2454276, -0.9865263, -2.2457206, -0.9858081, -0.9257988, 0.9256067
3: -5.1667862, -2.8748765, -5.1691270, -2.8740532, -1.2696269, 1.2715986
4: -2.7066915, -1.1295490, -2.7068958, -1.1290845, -0.6507539, 0.6506006
5: -5.5018702, -3.0855293, -5.5053544, -3.0844533, -1.3821687, 1.3851591
6: -5.3744178, -3.5245509, -5.3746614, -3.5238209, -0.4272761, 0.4265629
7: -4.6325011, -2.3180947, -4.6394134, -2.3165607, -1.4275435, 1.4350194
8: 0.3594735, 1.0434575, 0.3588101, 1.0467484, -0.5427969, 0.5395145
9: -1.3550586, 0.1109477, -1.3567817, 0.1114554, -1.3170011, 1.3182291

Time for backsubstitution: 5.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3442
type: B, layer: 1, pos: 380
type: B, layer: 1, pos: 3444
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 3412
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 3408
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3422
type: B, layer: 1, pos: 3423
type: B, layer: 1, pos: 3445
type: B, layer: 1, pos: 483
type: B, layer: 1, pos: 3425
type: B, layer: 1, pos: 3409
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3410
type: B, layer: 1, pos: 3414
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3411
type: B, layer: 1, pos: 3424
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 3430
type: B, layer: 1, pos: 504
type: B, layer: 1, pos: 486
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 3436
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 503
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 3460
type: B, layer: 1, pos: 469
type: B, layer: 1, pos: 2432
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 499
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 2223
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 3429
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 3452
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2152
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 3451
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 2170
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 3496
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 2651
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 507
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 2682
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 3490
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 3465
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 3482
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 3503
type: B, layer: 1, pos: 3467
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3250
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 432
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 3474
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 3495
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 3466
type: B, layer: 1, pos: 3481
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 435
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2697

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3442

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1342155, upper bound: 0.1342236
time: 377.26 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1342155, upper bound: 0.1343389
time: 263.24 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.2663592, 0.3793317, -0.2663658, 0.3793319, -0.4738953, 0.4741135
1: -1.4844741, -0.3354808, -1.4844742, -0.3354801, -0.4965171, 0.4961793
2: -2.2458715, -0.9846599, -2.2458735, -0.9846502, -0.9277232, 0.9277011
3: -5.1692162, -2.8713934, -5.1692176, -2.8713696, -1.2751656, 1.2748863
4: -2.7080133, -1.1290675, -2.7080135, -1.1290673, -0.6518708, 0.6518852
5: -5.5053792, -3.0807824, -5.5053806, -3.0807507, -1.3900985, 1.3897104
6: -5.3754110, -3.5238078, -5.3754177, -3.5238082, -0.4283068, 0.4283506
7: -4.6394367, -2.3095491, -4.6394372, -2.3094893, -1.4446137, 1.4438922
8: 0.3552409, 1.0467502, 0.3552139, 1.0467503, -0.5470576, 0.5474201
9: -1.3568649, 0.1130996, -1.3568655, 0.1131150, -1.3204763, 1.3202050

Time for backsubstitution: 5.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3442
type: B, layer: 1, pos: 380
type: B, layer: 1, pos: 3444
type: B, layer: 1, pos: 3427
type: B, layer: 1, pos: 3428
type: B, layer: 1, pos: 3413
type: B, layer: 1, pos: 3412
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 3446
type: B, layer: 1, pos: 3408
type: B, layer: 1, pos: 489
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 3422
type: B, layer: 1, pos: 3445
type: B, layer: 1, pos: 3423
type: B, layer: 1, pos: 483
type: B, layer: 1, pos: 3425
type: B, layer: 1, pos: 3409
type: B, layer: 1, pos: 487
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 522
type: B, layer: 1, pos: 3414
type: B, layer: 1, pos: 3410
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 3411
type: B, layer: 1, pos: 3424
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 485
type: B, layer: 1, pos: 521
type: B, layer: 1, pos: 523
type: B, layer: 1, pos: 506
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 3438
type: B, layer: 1, pos: 504
type: B, layer: 1, pos: 3430
type: B, layer: 1, pos: 486
type: B, layer: 1, pos: 505
type: B, layer: 1, pos: 517
type: B, layer: 1, pos: 490
type: B, layer: 1, pos: 3461
type: B, layer: 1, pos: 3462
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 3436
type: B, layer: 1, pos: 484
type: B, layer: 1, pos: 503
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 473
type: B, layer: 1, pos: 2433
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 3439
type: B, layer: 1, pos: 474
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 500
type: B, layer: 1, pos: 472
type: B, layer: 1, pos: 470
type: B, layer: 1, pos: 3460
type: B, layer: 1, pos: 469
type: B, layer: 1, pos: 2432
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 499
type: B, layer: 1, pos: 2193
type: B, layer: 1, pos: 537
type: B, layer: 1, pos: 3429
type: B, layer: 1, pos: 2223
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 3440
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 2226
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 513
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 3452
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 519
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 3497
type: B, layer: 1, pos: 2152
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 3451
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 471
type: B, layer: 1, pos: 573
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 2647
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 3124
type: B, layer: 1, pos: 143
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2195
type: B, layer: 1, pos: 572
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 2151
type: B, layer: 1, pos: 3231
type: B, layer: 1, pos: 2606
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 2170
type: B, layer: 1, pos: 577
type: B, layer: 1, pos: 3496
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 3053
type: B, layer: 1, pos: 2614
type: B, layer: 1, pos: 3453
type: B, layer: 1, pos: 3055
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 2651
type: B, layer: 1, pos: 2163
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2388
type: B, layer: 1, pos: 3516
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 2169
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 507
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 2682
type: B, layer: 1, pos: 3475
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2569
type: B, layer: 1, pos: 510
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 3490
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 3450
type: B, layer: 1, pos: 571
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 3245
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 535
type: B, layer: 1, pos: 3465
type: B, layer: 1, pos: 849
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 3482
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 2346
type: B, layer: 1, pos: 3503
type: B, layer: 1, pos: 3467
type: B, layer: 1, pos: 525
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 3250
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 555
type: B, layer: 1, pos: 432
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 850
type: B, layer: 1, pos: 371
type: B, layer: 1, pos: 526
type: B, layer: 1, pos: 527
type: B, layer: 1, pos: 3474
type: B, layer: 1, pos: 2140
type: B, layer: 1, pos: 556
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 528
type: B, layer: 1, pos: 558
type: B, layer: 1, pos: 541
type: B, layer: 1, pos: 3495
type: B, layer: 1, pos: 550
type: B, layer: 1, pos: 543
type: B, layer: 1, pos: 3466
type: B, layer: 1, pos: 2123
type: B, layer: 1, pos: 3481
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 3468
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 435
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 2460
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2697

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3442

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1342155, upper bound: 0.1342256
time: 259.03 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1342155, upper bound: 0.1342239
time: 59.07 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 324.13 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 324.13
Output dim: 8, lower bound: -0.1342155, upper bound: 0.1342236
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 324.13
Output dim: 8, lower bound: -0.1342155, upper bound: 0.1343389
NS_A2_B1, status: Status.VERIFIED, split count: 2, time: 324.13
Output dim: 8, lower bound: -0.1342155, upper bound: 0.1342256
NS_A2_B2, status: Status.VERIFIED, split count: 2, time: 324.13
Output dim: 8, lower bound: -0.1342155, upper bound: 0.1342239

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 301.28 + 1599.32 = 1900.59 seconds
