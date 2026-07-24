## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 1)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.0663481378


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7154022, 0.7154022)
1: (-4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7588421, 0.7588422)
2: (-1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776742, 0.4776741)
3: (-0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1819811, 0.1819811)
4: (-1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1645983, 0.1645983)
5: (-0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041713, 0.1041713)
6: (-2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3511744, 0.3511744)
7: (0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065590, 0.4065590)
8: (-5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4574489, 0.4574490)
9: (-3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7732574, 0.7732574)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.57 + 167.09 = 174.65 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0664757, upper bound: 0.0664833

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 482

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 509

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664736, upper bound: 0.0664844
time: 31.74 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664736, upper bound: 0.0664841
time: 19.21 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 50.96 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 50.96
Output dim: 5, lower bound: -0.0664736, upper bound: 0.0664844
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 50.96
Output dim: 5, lower bound: -0.0664736, upper bound: 0.0664841

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7154022, 0.7154022
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7588421, 0.7588422
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776742, 0.4776741
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1819811, 0.1819811
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1645983, 0.1645983
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041713, 0.1041713
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3511744, 0.3511744
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065590, 0.4065590
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4574489, 0.4574490
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7732574, 0.7732574

Time for backsubstitution: 5.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3435

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3556

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664753, upper bound: 0.0664723
time: 9.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664601, upper bound: 0.0664799
time: 16.24 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7154022, 0.7154022
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7588421, 0.7588422
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776742, 0.4776741
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1819811, 0.1819811
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1645983, 0.1645983
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041713, 0.1041713
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3511744, 0.3511744
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065590, 0.4065590
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4574489, 0.4574490
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7732574, 0.7732574

Time for backsubstitution: 5.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2061

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664464, upper bound: 0.0664797
time: 79.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664740, upper bound: 0.0664539
time: 113.90 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 199.01 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 199.01
Output dim: 5, lower bound: -0.0664753, upper bound: 0.0664723
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 199.01
Output dim: 5, lower bound: -0.0664601, upper bound: 0.0664799
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 199.01
Output dim: 5, lower bound: -0.0664464, upper bound: 0.0664797
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 199.01
Output dim: 5, lower bound: -0.0664740, upper bound: 0.0664539

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7152618, 0.7152886
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7581534, 0.7580551
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776432, 0.4776473
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1811303, 0.1810012
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1644353, 0.1644571
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1040943, 0.1040842
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3496012, 0.3493577
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065179, 0.4065080
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4573846, 0.4573781
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7723948, 0.7722629

Time for backsubstitution: 5.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 770

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 698

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664742, upper bound: 0.0664686
time: 44.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664720, upper bound: 0.0664713
time: 9.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7152886, 0.7152618
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7580550, 0.7581534
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776473, 0.4776431
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1810012, 0.1811302
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1644571, 0.1644352
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1040842, 0.1040943
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3493577, 0.3496012
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065080, 0.4065179
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4573781, 0.4573846
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7722628, 0.7723950

Time for backsubstitution: 5.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3523

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2450

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664628, upper bound: 0.0664790
time: 56.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664642, upper bound: 0.0664845
time: 9.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7151325, 0.7151111
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7556577, 0.7554190
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776705, 0.4776696
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1818454, 0.1818540
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1646055, 0.1646030
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1040775, 0.1040812
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3504391, 0.3504563
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065502, 0.4065486
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4571739, 0.4571590
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7716744, 0.7715828

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2393

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 194

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664494, upper bound: 0.0664778
time: 308.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664494, upper bound: 0.0664770
time: 67.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7151111, 0.7151326
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7554189, 0.7556577
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776696, 0.4776705
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1818540, 0.1818454
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1646030, 0.1646055
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1040812, 0.1040775
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3504563, 0.3504391
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065486, 0.4065502
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4571590, 0.4571739
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7715828, 0.7716744

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 3577

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3421

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664244, upper bound: 0.0664548
time: 84.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664749, upper bound: 0.0664122
time: 40.97 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 131.79 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 131.79
Output dim: 5, lower bound: -0.0664742, upper bound: 0.0664686
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 131.79
Output dim: 5, lower bound: -0.0664720, upper bound: 0.0664713
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 131.79
Output dim: 5, lower bound: -0.0664628, upper bound: 0.0664790
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 131.79
Output dim: 5, lower bound: -0.0664642, upper bound: 0.0664845
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 131.79
Output dim: 5, lower bound: -0.0664494, upper bound: 0.0664778
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 131.79
Output dim: 5, lower bound: -0.0664494, upper bound: 0.0664770
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 131.79
Output dim: 5, lower bound: -0.0664244, upper bound: 0.0664548
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 131.79
Output dim: 5, lower bound: -0.0664749, upper bound: 0.0664122

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7152616, 0.7152885
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7581262, 0.7580400
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776421, 0.4776462
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1811299, 0.1810008
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1644349, 0.1644566
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1040939, 0.1040838
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3495985, 0.3493548
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065176, 0.4065077
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4573796, 0.4573733
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7723725, 0.7722414

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2524

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3521

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664713, upper bound: 0.0664304
time: 54.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664360, upper bound: 0.0664667
time: 12.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7152616, 0.7152885
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7581382, 0.7580279
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776421, 0.4776463
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1811299, 0.1810008
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1644347, 0.1644568
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1040939, 0.1040838
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3495983, 0.3493550
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065176, 0.4065077
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4573798, 0.4573731
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7723735, 0.7722405

Time for backsubstitution: 5.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2947

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3420

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664704, upper bound: 0.0664685
time: 11.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664709, upper bound: 0.0664687
time: 73.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7152830, 0.7152561
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7580419, 0.7581401
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776469, 0.4776428
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1809997, 0.1811288
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1644571, 0.1644352
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1040840, 0.1040941
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3493572, 0.3496007
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065076, 0.4065174
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4573660, 0.4573722
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7722570, 0.7723885

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3491

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664577, upper bound: 0.0663574
time: 7.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663326, upper bound: 0.0664804
time: 8.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7152829, 0.7152562
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7580417, 0.7581403
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776469, 0.4776428
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1809997, 0.1811288
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1644571, 0.1644352
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1040839, 0.1040941
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3493572, 0.3496007
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065076, 0.4065174
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4573658, 0.4573725
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7722564, 0.7723891

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3420

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3523

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664641, upper bound: 0.0664772
time: 54.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664588, upper bound: 0.0664781
time: 31.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7151325, 0.7151111
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7556577, 0.7554190
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776705, 0.4776696
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1818454, 0.1818540
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1646055, 0.1646030
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1040775, 0.1040812
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3504391, 0.3504563
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065502, 0.4065486
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4571739, 0.4571590
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7716744, 0.7715828

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 401

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664461, upper bound: 0.0664368
time: 7.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664018, upper bound: 0.0664057
time: 309.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7151325, 0.7151111
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7556577, 0.7554190
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776705, 0.4776696
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1818454, 0.1818540
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1646055, 0.1646030
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1040775, 0.1040812
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3504391, 0.3504563
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065502, 0.4065486
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4571739, 0.4571590
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7716744, 0.7715828

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 497

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 636

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664371, upper bound: 0.0664789
time: 41.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664449, upper bound: 0.0664711
time: 26.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7162877, 0.7162522
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7543790, 0.7545668
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4772320, 0.4772523
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1815219, 0.1815426
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1644948, 0.1644540
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1033383, 0.1033739
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3507893, 0.3508077
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4067219, 0.4067278
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4567452, 0.4567515
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7714124, 0.7715031

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 3435

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663993, upper bound: 0.0664539
time: 10.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664243, upper bound: 0.0664236
time: 63.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7162308, 0.7163091
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7543279, 0.7546178
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4772515, 0.4772329
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1815511, 0.1815133
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1644514, 0.1644974
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1033777, 0.1033345
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3508248, 0.3507722
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4067261, 0.4067235
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4567367, 0.4567601
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7714115, 0.7715040

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3353

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3308

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664740, upper bound: 0.0663825
time: 5.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664384, upper bound: 0.0664103
time: 29.47 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 41.22 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 41.22
Output dim: 5, lower bound: -0.0664713, upper bound: 0.0664304
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 41.22
Output dim: 5, lower bound: -0.0664360, upper bound: 0.0664667
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 41.22
Output dim: 5, lower bound: -0.0664704, upper bound: 0.0664685
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 41.22
Output dim: 5, lower bound: -0.0664709, upper bound: 0.0664687
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 41.22
Output dim: 5, lower bound: -0.0664577, upper bound: 0.0663574
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 41.22
Output dim: 5, lower bound: -0.0663326, upper bound: 0.0664804
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 41.22
Output dim: 5, lower bound: -0.0664641, upper bound: 0.0664772
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 41.22
Output dim: 5, lower bound: -0.0664588, upper bound: 0.0664781
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 41.22
Output dim: 5, lower bound: -0.0664461, upper bound: 0.0664368
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 41.22
Output dim: 5, lower bound: -0.0664018, upper bound: 0.0664057
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 41.22
Output dim: 5, lower bound: -0.0664371, upper bound: 0.0664789
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 41.22
Output dim: 5, lower bound: -0.0664449, upper bound: 0.0664711
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 41.22
Output dim: 5, lower bound: -0.0663993, upper bound: 0.0664539
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 41.22
Output dim: 5, lower bound: -0.0664243, upper bound: 0.0664236
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 41.22
Output dim: 5, lower bound: -0.0664740, upper bound: 0.0663825
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 41.22
Output dim: 5, lower bound: -0.0664384, upper bound: 0.0664103

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7143637, 0.7144384
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7581794, 0.7580981
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776444, 0.4776486
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1805401, 0.1803846
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1641802, 0.1642123
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1038688, 0.1038502
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3468192, 0.3464403
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4064205, 0.4064151
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4573973, 0.4574536
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7718130, 0.7716560

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 3289

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2303

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664575, upper bound: 0.0664076
time: 11.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664417, upper bound: 0.0664219
time: 8.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7144115, 0.7143905
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7581844, 0.7580932
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776445, 0.4776485
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1805137, 0.1804110
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1641906, 0.1642019
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1038603, 0.1038586
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3466840, 0.3465754
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4064249, 0.4064106
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4574599, 0.4573910
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7717872, 0.7716818

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 3025

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2942

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664196, upper bound: 0.0664559
time: 128.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664230, upper bound: 0.0664574
time: 6.91 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7152346, 0.7152534
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7581351, 0.7580251
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776301, 0.4776307
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1811190, 0.1809936
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1644038, 0.1644199
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1040887, 0.1040798
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3495903, 0.3493502
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065127, 0.4065022
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4573235, 0.4573021
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7723543, 0.7722257

Time for backsubstitution: 6.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 774

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3287

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664739, upper bound: 0.0664681
time: 7.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664663, upper bound: 0.0664718
time: 10.11 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7152265, 0.7152615
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7581355, 0.7580247
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776265, 0.4776343
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1811226, 0.1809899
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643979, 0.1644258
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1040899, 0.1040786
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3495935, 0.3493469
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065121, 0.4065028
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4573089, 0.4573167
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7723587, 0.7722213

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 786

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 784

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664662, upper bound: 0.0664694
time: 30.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664731, upper bound: 0.0664599
time: 55.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7151782, 0.7151716
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7579715, 0.7581017
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4775929, 0.4775231
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1808741, 0.1809580
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643353, 0.1643458
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1039523, 0.1039138
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3490315, 0.3491632
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4063628, 0.4064168
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4572343, 0.4572937
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7722563, 0.7723878

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 3288

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2363

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663377, upper bound: 0.0663448
time: 11.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664491, upper bound: 0.0662316
time: 33.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7151986, 0.7151511
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7580035, 0.7580698
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4775272, 0.4775888
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1808289, 0.1810032
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643677, 0.1643133
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1039037, 0.1039624
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3489197, 0.3492750
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4064069, 0.4063727
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4572874, 0.4572406
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7722561, 0.7723878

Time for backsubstitution: 6.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3557

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2524

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663231, upper bound: 0.0664684
time: 6.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663239, upper bound: 0.0664674
time: 7.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7152815, 0.7152551
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7580390, 0.7581377
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776467, 0.4776425
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1809993, 0.1811284
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1644568, 0.1644349
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1040832, 0.1040935
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3493567, 0.3496003
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065065, 0.4065163
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4573643, 0.4573712
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7722557, 0.7723883

Time for backsubstitution: 6.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2675

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 566

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664503, upper bound: 0.0663392
time: 48.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663288, upper bound: 0.0664741
time: 9.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7152818, 0.7152549
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7580392, 0.7581375
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776467, 0.4776425
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1809994, 0.1811284
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1644567, 0.1644349
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1040833, 0.1040934
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3493568, 0.3496003
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065064, 0.4065164
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4573645, 0.4573711
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7722557, 0.7723883

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 3200

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 743

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664403, upper bound: 0.0664781
time: 51.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664620, upper bound: 0.0664584
time: 135.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7166125, 0.7167044
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7506321, 0.7506533
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776117, 0.4776077
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1777913, 0.1780039
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1636114, 0.1635568
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1042653, 0.1042836
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3437969, 0.3441402
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4044745, 0.4043645
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4555677, 0.4556977
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7654889, 0.7657382

Time for backsubstitution: 6.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3435

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 851

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664430, upper bound: 0.0664325
time: 45.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664421, upper bound: 0.0664324
time: 12.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7167258, 0.7165911
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7508921, 0.7503934
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776086, 0.4776108
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1779954, 0.1777998
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1635593, 0.1636088
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1042798, 0.1042691
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3441231, 0.3438141
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4043661, 0.4044729
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4557126, 0.4555528
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7658297, 0.7653973

Time for backsubstitution: 6.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3420

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2941

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663895, upper bound: 0.0663946
time: 39.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663905, upper bound: 0.0664653
time: 128.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7150927, 0.7150694
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7554753, 0.7552235
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776416, 0.4776421
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1816324, 0.1816516
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1645901, 0.1645841
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1040630, 0.1040674
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3504173, 0.3504329
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065480, 0.4065463
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4571666, 0.4571516
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7716308, 0.7715428

Time for backsubstitution: 6.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 2545

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 552

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664348, upper bound: 0.0664614
time: 55.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664163, upper bound: 0.0664768
time: 42.09 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7150908, 0.7150714
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7554623, 0.7552365
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776430, 0.4776407
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1816430, 0.1816409
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1645867, 0.1645875
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1040637, 0.1040668
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3504158, 0.3504345
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065479, 0.4065463
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4571665, 0.4571517
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7716344, 0.7715392

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 3421

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3116

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664339, upper bound: 0.0664615
time: 19.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664411, upper bound: 0.0664554
time: 50.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7162967, 0.7162615
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7543886, 0.7545767
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4772324, 0.4772527
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1815233, 0.1815441
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1644938, 0.1644530
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1033391, 0.1033747
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3507828, 0.3508014
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4067463, 0.4067512
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4567598, 0.4567662
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7714099, 0.7715005

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 596

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2545

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663907, upper bound: 0.0664176
time: 61.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663650, upper bound: 0.0664481
time: 42.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7162969, 0.7162613
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7543889, 0.7545765
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4772324, 0.4772527
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1815234, 0.1815440
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1644938, 0.1644530
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1033391, 0.1033747
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3507831, 0.3508011
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4067454, 0.4067521
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4567599, 0.4567661
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7714099, 0.7715005

Time for backsubstitution: 6.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 3542

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3116

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664113, upper bound: 0.0664198
time: 8.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664174, upper bound: 0.0664104
time: 9.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7164121, 0.7164893
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7543445, 0.7546358
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4772503, 0.4772318
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1813769, 0.1813284
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1645063, 0.1645591
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1033397, 0.1032942
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3506858, 0.3506155
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4067452, 0.4067431
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4569309, 0.4569523
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7714680, 0.7715641

Time for backsubstitution: 6.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3256

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3493

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664732, upper bound: 0.0663101
time: 20.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664019, upper bound: 0.0663814
time: 12.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7164111, 0.7164903
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7543458, 0.7546344
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4772503, 0.4772318
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1813662, 0.1813391
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1645131, 0.1645523
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1033373, 0.1032965
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3506682, 0.3506331
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4067456, 0.4067426
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4569288, 0.4569544
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7714715, 0.7715605

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2488

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 613

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664420, upper bound: 0.0664071
time: 10.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664339, upper bound: 0.0664108
time: 10.14 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 27.11 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0664575, upper bound: 0.0664076
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0664417, upper bound: 0.0664219
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0664196, upper bound: 0.0664559
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0664230, upper bound: 0.0664574
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0664739, upper bound: 0.0664681
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0664663, upper bound: 0.0664718
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0664662, upper bound: 0.0664694
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0664731, upper bound: 0.0664599
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0663377, upper bound: 0.0663448
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0664491, upper bound: 0.0662316
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0663231, upper bound: 0.0664684
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0663239, upper bound: 0.0664674
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0664503, upper bound: 0.0663392
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0663288, upper bound: 0.0664741
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0664403, upper bound: 0.0664781
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0664620, upper bound: 0.0664584
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0664430, upper bound: 0.0664325
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0664421, upper bound: 0.0664324
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0663895, upper bound: 0.0663946
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0663905, upper bound: 0.0664653
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0664348, upper bound: 0.0664614
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0664163, upper bound: 0.0664768
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0664339, upper bound: 0.0664615
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0664411, upper bound: 0.0664554
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0663907, upper bound: 0.0664176
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0663650, upper bound: 0.0664481
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0664113, upper bound: 0.0664198
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0664174, upper bound: 0.0664104
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0664732, upper bound: 0.0663101
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0664019, upper bound: 0.0663814
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0664420, upper bound: 0.0664071
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 27.11
Output dim: 5, lower bound: -0.0664339, upper bound: 0.0664108

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7135305, 0.7135821
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7518359, 0.7515562
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4775870, 0.4775927
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1803977, 0.1802416
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1641831, 0.1642155
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1037553, 0.1037363
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3455154, 0.3451805
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4064192, 0.4064135
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4569756, 0.4570389
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7700154, 0.7697509

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2393

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 830

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664568, upper bound: 0.0664063
time: 7.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664592, upper bound: 0.0664074
time: 10.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7135073, 0.7136053
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7516376, 0.7517545
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4775884, 0.4775913
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1803972, 0.1802423
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1641833, 0.1642153
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1037549, 0.1037367
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3455593, 0.3451366
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4064189, 0.4064137
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4569826, 0.4570320
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7699078, 0.7698584

Time for backsubstitution: 6.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 848

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 463

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663803, upper bound: 0.0664091
time: 9.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664305, upper bound: 0.0663566
time: 51.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7143511, 0.7143271
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7581521, 0.7580637
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776402, 0.4776432
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1805101, 0.1804078
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1641825, 0.1641881
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1038566, 0.1038553
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3466443, 0.3465373
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4064202, 0.4064057
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4572984, 0.4572580
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7717841, 0.7716794

Time for backsubstitution: 6.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2530

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3371

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664169, upper bound: 0.0664540
time: 8.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664218, upper bound: 0.0664556
time: 12.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7143480, 0.7143302
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7581547, 0.7580611
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776393, 0.4776442
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1805105, 0.1804074
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1641767, 0.1641938
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1038569, 0.1038549
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3466459, 0.3465358
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4064200, 0.4064059
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4573269, 0.4572296
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7717848, 0.7716787

Time for backsubstitution: 6.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 3353

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664156, upper bound: 0.0664476
time: 10.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664171, upper bound: 0.0664115
time: 111.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7152301, 0.7152393
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7580559, 0.7579691
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776211, 0.4776145
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1810997, 0.1809623
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643865, 0.1644093
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1040820, 0.1040690
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3495738, 0.3493431
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4064239, 0.4064501
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4572892, 0.4572461
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7723518, 0.7722229

Time for backsubstitution: 6.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 17

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3552

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664730, upper bound: 0.0664527
time: 36.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664583, upper bound: 0.0664676
time: 7.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7152205, 0.7152489
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7580789, 0.7579460
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776139, 0.4776217
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1810876, 0.1809743
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643932, 0.1644026
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1040779, 0.1040731
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3495831, 0.3493337
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4064607, 0.4064134
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4572675, 0.4572679
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7723516, 0.7722231

Time for backsubstitution: 6.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 3598
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 3557

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3523

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664719, upper bound: 0.0664698
time: 10.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664674, upper bound: 0.0664676
time: 171.44 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 187.90 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 187.90
Output dim: 5, lower bound: -0.0664568, upper bound: 0.0664063
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 187.90
Output dim: 5, lower bound: -0.0664592, upper bound: 0.0664074
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 187.90
Output dim: 5, lower bound: -0.0663803, upper bound: 0.0664091
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 187.90
Output dim: 5, lower bound: -0.0664305, upper bound: 0.0663566
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 187.90
Output dim: 5, lower bound: -0.0664169, upper bound: 0.0664540
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 187.90
Output dim: 5, lower bound: -0.0664218, upper bound: 0.0664556
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 187.90
Output dim: 5, lower bound: -0.0664156, upper bound: 0.0664476
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 187.90
Output dim: 5, lower bound: -0.0664171, upper bound: 0.0664115
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 187.90
Output dim: 5, lower bound: -0.0664730, upper bound: 0.0664527
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 187.90
Output dim: 5, lower bound: -0.0664583, upper bound: 0.0664676
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 187.90
Output dim: 5, lower bound: -0.0664719, upper bound: 0.0664698
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 187.90
Output dim: 5, lower bound: -0.0664674, upper bound: 0.0664676
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 187.90
Output dim: 5, lower bound: -0.0664662, upper bound: 0.0664694
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 187.90
Output dim: 5, lower bound: -0.0664731, upper bound: 0.0664599
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 187.90
Output dim: 5, lower bound: -0.0664491, upper bound: 0.0662316
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 187.90
Output dim: 5, lower bound: -0.0663231, upper bound: 0.0664684
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 187.90
Output dim: 5, lower bound: -0.0663239, upper bound: 0.0664674
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 187.90
Output dim: 5, lower bound: -0.0664503, upper bound: 0.0663392
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 187.90
Output dim: 5, lower bound: -0.0663288, upper bound: 0.0664741
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 187.90
Output dim: 5, lower bound: -0.0664403, upper bound: 0.0664781
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 187.90
Output dim: 5, lower bound: -0.0664620, upper bound: 0.0664584
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 187.90
Output dim: 5, lower bound: -0.0664430, upper bound: 0.0664325
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 187.90
Output dim: 5, lower bound: -0.0664421, upper bound: 0.0664324
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 187.90
Output dim: 5, lower bound: -0.0663895, upper bound: 0.0663946
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 187.90
Output dim: 5, lower bound: -0.0663905, upper bound: 0.0664653
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 187.90
Output dim: 5, lower bound: -0.0664348, upper bound: 0.0664614
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 187.90
Output dim: 5, lower bound: -0.0664163, upper bound: 0.0664768
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 187.90
Output dim: 5, lower bound: -0.0664339, upper bound: 0.0664615
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 187.90
Output dim: 5, lower bound: -0.0664411, upper bound: 0.0664554
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 187.90
Output dim: 5, lower bound: -0.0663907, upper bound: 0.0664176
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 187.90
Output dim: 5, lower bound: -0.0663650, upper bound: 0.0664481
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 187.90
Output dim: 5, lower bound: -0.0664113, upper bound: 0.0664198
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 187.90
Output dim: 5, lower bound: -0.0664174, upper bound: 0.0664104
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 187.90
Output dim: 5, lower bound: -0.0664732, upper bound: 0.0663101
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 187.90
Output dim: 5, lower bound: -0.0664019, upper bound: 0.0663814
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 187.90
Output dim: 5, lower bound: -0.0664420, upper bound: 0.0664071
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 187.90
Output dim: 5, lower bound: -0.0664339, upper bound: 0.0664108

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 174.65 + 3434.69 = 3609.35 seconds
