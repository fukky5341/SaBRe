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
execution time: IAR + RelationalAnalysis = 7.80 + 294.95 = 302.75 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.1343695, upper bound: 0.1343626

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 3424
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 3442
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 3429
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 432
type: DSZ, layer: 1, pos: 3409
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 485
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3408
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 3414
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3411
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 504
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3410
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 519
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 469
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 571
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 3412
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3495
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3503
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 555
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 435
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 3516

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3477

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1343413, upper bound: 0.1343576
time: 51.65 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1343668, upper bound: 0.1343355
time: 469.54 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 521.27 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 521.27
Output dim: 8, lower bound: -0.1343413, upper bound: 0.1343576
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 521.27
Output dim: 8, lower bound: -0.1343668, upper bound: 0.1343355

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.2664312, 0.3793331, -0.2664312, 0.3793331, -0.4741472, 0.4741471
1: -1.4844748, -0.3354728, -1.4844748, -0.3354728, -0.4965301, 0.4965301
2: -2.2458954, -0.9845571, -2.2458954, -0.9845571, -0.9277976, 0.9277976
3: -5.1692300, -2.8711410, -5.1692300, -2.8711410, -1.2754008, 1.2754008
4: -2.7080197, -1.1290655, -2.7080197, -1.1290655, -0.6519032, 0.6519033
5: -5.5053887, -3.0804513, -5.5053887, -3.0804513, -1.3902578, 1.3902578
6: -5.3754797, -3.5238063, -5.3754797, -3.5238063, -0.4283697, 0.4283697
7: -4.6394372, -2.3089204, -4.6394372, -2.3089204, -1.4446172, 1.4446173
8: 0.3549576, 1.0467505, 0.3549576, 1.0467505, -0.5474604, 0.5474604
9: -1.3568701, 0.1132612, -1.3568701, 0.1132612, -1.3206248, 1.3206248

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 3424
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 3442
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 3429
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 432
type: DSZ, layer: 1, pos: 3409
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 485
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3408
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 3414
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3411
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 504
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3410
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 519
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 469
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 571
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 3412
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3495
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3503
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 555
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 435
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 3516

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3462

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1342785, upper bound: 0.1343606
time: 188.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1343408, upper bound: 0.1342708
time: 494.44 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.2664312, 0.3793331, -0.2664312, 0.3793331, -0.4741471, 0.4741472
1: -1.4844748, -0.3354728, -1.4844748, -0.3354728, -0.4965301, 0.4965301
2: -2.2458954, -0.9845571, -2.2458954, -0.9845571, -0.9277976, 0.9277976
3: -5.1692300, -2.8711410, -5.1692300, -2.8711410, -1.2754008, 1.2754009
4: -2.7080197, -1.1290655, -2.7080197, -1.1290655, -0.6519033, 0.6519032
5: -5.5053887, -3.0804513, -5.5053887, -3.0804513, -1.3902578, 1.3902578
6: -5.3754797, -3.5238063, -5.3754797, -3.5238063, -0.4283697, 0.4283697
7: -4.6394372, -2.3089204, -4.6394372, -2.3089204, -1.4446172, 1.4446173
8: 0.3549576, 1.0467505, 0.3549576, 1.0467505, -0.5474604, 0.5474605
9: -1.3568701, 0.1132612, -1.3568701, 0.1132612, -1.3206249, 1.3206247

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3462
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 3424
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 3442
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 3429
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 432
type: DSZ, layer: 1, pos: 3409
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 485
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3408
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 3414
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3411
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 504
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3410
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 519
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 469
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 571
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 3412
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3495
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3503
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 555
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 435
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 3516

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3462

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1342845, upper bound: 0.1343322
time: 23.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1343596, upper bound: 0.1342723
time: 47.13 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 76.50 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 76.50
Output dim: 8, lower bound: -0.1342785, upper bound: 0.1343606
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 76.50
Output dim: 8, lower bound: -0.1343408, upper bound: 0.1342708
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 76.50
Output dim: 8, lower bound: -0.1342845, upper bound: 0.1343322
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 76.50
Output dim: 8, lower bound: -0.1343596, upper bound: 0.1342723

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.2664312, 0.3793331, -0.2664312, 0.3793331, -0.4741454, 0.4741453
1: -1.4844748, -0.3354728, -1.4844748, -0.3354728, -0.4965194, 0.4965195
2: -2.2458954, -0.9845571, -2.2458954, -0.9845571, -0.9277970, 0.9277971
3: -5.1692300, -2.8711410, -5.1692300, -2.8711410, -1.2754021, 1.2754019
4: -2.7080197, -1.1290655, -2.7080197, -1.1290655, -0.6518854, 0.6518856
5: -5.5053887, -3.0804513, -5.5053887, -3.0804513, -1.3902574, 1.3902574
6: -5.3754797, -3.5238063, -5.3754797, -3.5238063, -0.4283659, 0.4283659
7: -4.6394372, -2.3089204, -4.6394372, -2.3089204, -1.4446104, 1.4446104
8: 0.3549576, 1.0467505, 0.3549576, 1.0467505, -0.5474610, 0.5474610
9: -1.3568701, 0.1132612, -1.3568701, 0.1132612, -1.3206203, 1.3206202

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3446
type: DSZ, layer: 1, pos: 3425
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3430
type: DSZ, layer: 1, pos: 3440
type: DSZ, layer: 1, pos: 3445
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3461
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 523
type: DSZ, layer: 1, pos: 3424
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 3439
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 522
type: DSZ, layer: 1, pos: 3442
type: DSZ, layer: 1, pos: 506
type: DSZ, layer: 1, pos: 537
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 3423
type: DSZ, layer: 1, pos: 3444
type: DSZ, layer: 1, pos: 3429
type: DSZ, layer: 1, pos: 3438
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 500
type: DSZ, layer: 1, pos: 521
type: DSZ, layer: 1, pos: 505
type: DSZ, layer: 1, pos: 517
type: DSZ, layer: 1, pos: 507
type: DSZ, layer: 1, pos: 490
type: DSZ, layer: 1, pos: 499
type: DSZ, layer: 1, pos: 432
type: DSZ, layer: 1, pos: 3409
type: DSZ, layer: 1, pos: 3482
type: DSZ, layer: 1, pos: 485
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 472
type: DSZ, layer: 1, pos: 3497
type: DSZ, layer: 1, pos: 487
type: DSZ, layer: 1, pos: 3422
type: DSZ, layer: 1, pos: 3408
type: DSZ, layer: 1, pos: 471
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 484
type: DSZ, layer: 1, pos: 3414
type: DSZ, layer: 1, pos: 486
type: DSZ, layer: 1, pos: 498
type: DSZ, layer: 1, pos: 3460
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 3481
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 3467
type: DSZ, layer: 1, pos: 558
type: DSZ, layer: 1, pos: 536
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 3436
type: DSZ, layer: 1, pos: 513
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 543
type: DSZ, layer: 1, pos: 3250
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 573
type: DSZ, layer: 1, pos: 3490
type: DSZ, layer: 1, pos: 542
type: DSZ, layer: 1, pos: 470
type: DSZ, layer: 1, pos: 3453
type: DSZ, layer: 1, pos: 3411
type: DSZ, layer: 1, pos: 572
type: DSZ, layer: 1, pos: 504
type: DSZ, layer: 1, pos: 3468
type: DSZ, layer: 1, pos: 3410
type: DSZ, layer: 1, pos: 512
type: DSZ, layer: 1, pos: 3466
type: DSZ, layer: 1, pos: 519
type: DSZ, layer: 1, pos: 3475
type: DSZ, layer: 1, pos: 489
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 3452
type: DSZ, layer: 1, pos: 3428
type: DSZ, layer: 1, pos: 371
type: DSZ, layer: 1, pos: 556
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 469
type: DSZ, layer: 1, pos: 3451
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 481
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 3474
type: DSZ, layer: 1, pos: 503
type: DSZ, layer: 1, pos: 2163
type: DSZ, layer: 1, pos: 571
type: DSZ, layer: 1, pos: 541
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 3413
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2195
type: DSZ, layer: 1, pos: 2170
type: DSZ, layer: 1, pos: 3412
type: DSZ, layer: 1, pos: 2606
type: DSZ, layer: 1, pos: 474
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 473
type: DSZ, layer: 1, pos: 535
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3495
type: DSZ, layer: 1, pos: 3055
type: DSZ, layer: 1, pos: 380
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 528
type: DSZ, layer: 1, pos: 3053
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3480
type: DSZ, layer: 1, pos: 3503
type: DSZ, layer: 1, pos: 2651
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2682
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 527
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 3450
type: DSZ, layer: 1, pos: 3231
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2169
type: DSZ, layer: 1, pos: 526
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 510
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 555
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 849
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 2388
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3124
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2123
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2432
type: DSZ, layer: 1, pos: 2614
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2346
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 2433
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 435
type: DSZ, layer: 1, pos: 444
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 577
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2460
type: DSZ, layer: 1, pos: 2569
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2697
type: DSZ, layer: 1, pos: 3245
type: DSZ, layer: 1, pos: 3516

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3446

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1341298, upper bound: 0.1342683
time: 368.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1342708, upper bound: 0.1342104
time: 19.95 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 394.15 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 394.15
Output dim: 8, lower bound: -0.1341298, upper bound: 0.1342683
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 394.15
Output dim: 8, lower bound: -0.1342708, upper bound: 0.1342104
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 394.15
Output dim: 8, lower bound: -0.1343408, upper bound: 0.1342708
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 394.15
Output dim: 8, lower bound: -0.1342845, upper bound: 0.1343322
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 394.15
Output dim: 8, lower bound: -0.1343596, upper bound: 0.1342723

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 302.75 + 1680.69 = 1983.44 seconds
