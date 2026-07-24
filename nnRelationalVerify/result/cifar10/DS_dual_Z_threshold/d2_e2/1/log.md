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
execution time: IAR + RelationalAnalysis = 8.37 + 161.15 = 169.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0664757, upper bound: 0.0664833

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2393

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664312, upper bound: 0.0664571
time: 133.17 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664518, upper bound: 0.0664398
time: 7.40 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 140.64 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 140.64
Output dim: 5, lower bound: -0.0664312, upper bound: 0.0664571
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 140.64
Output dim: 5, lower bound: -0.0664518, upper bound: 0.0664398

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7120891, 0.7120273
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7517705, 0.7516603
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776251, 0.4776348
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1818020, 0.1817931
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1644959, 0.1645013
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041428, 0.1041413
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3486954, 0.3485816
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065542, 0.4065542
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4558239, 0.4558895
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7725340, 0.7725067

Time for backsubstitution: 6.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3068

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663796, upper bound: 0.0664167
time: 7.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663836, upper bound: 0.0664054
time: 62.25 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7120273, 0.7120891
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7516603, 0.7517705
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4776349, 0.4776251
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1817931, 0.1818020
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1645013, 0.1644958
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041413, 0.1041428
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3485816, 0.3486954
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065542, 0.4065542
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4558895, 0.4558239
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7725066, 0.7725341

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3068
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 3068

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663991, upper bound: 0.0663963
time: 14.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664063, upper bound: 0.0663797
time: 40.37 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 61.60 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 61.60
Output dim: 5, lower bound: -0.0663796, upper bound: 0.0664167
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 61.60
Output dim: 5, lower bound: -0.0663836, upper bound: 0.0664054
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 61.60
Output dim: 5, lower bound: -0.0663991, upper bound: 0.0663963
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 61.60
Output dim: 5, lower bound: -0.0664063, upper bound: 0.0663797

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7088025, 0.7088168
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7433473, 0.7434300
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4773460, 0.4773602
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1816530, 0.1816431
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643795, 0.1643866
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041189, 0.1041163
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3453340, 0.3452041
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065500, 0.4065494
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4542964, 0.4544455
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7718763, 0.7718404

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2424

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663603, upper bound: 0.0663958
time: 17.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663624, upper bound: 0.0663973
time: 9.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7088786, 0.7087436
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7435402, 0.7432411
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4773504, 0.4773558
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1816520, 0.1816441
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643811, 0.1643850
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041178, 0.1041174
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3453187, 0.3452202
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065495, 0.4065500
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4543799, 0.4543621
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7718678, 0.7718487

Time for backsubstitution: 6.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2424

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663675, upper bound: 0.0663872
time: 55.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663659, upper bound: 0.0663826
time: 42.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7087437, 0.7088786
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7432411, 0.7435400
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4773558, 0.4773504
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1816441, 0.1816520
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643850, 0.1643811
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041174, 0.1041178
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3452203, 0.3453187
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065499, 0.4065495
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4543622, 0.4543798
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7718489, 0.7718678

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2424

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663817, upper bound: 0.0663772
time: 42.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663826, upper bound: 0.0663783
time: 18.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7088169, 0.7088025
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7434300, 0.7433473
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4773602, 0.4773460
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1816431, 0.1816530
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643866, 0.1643795
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041163, 0.1041189
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3452041, 0.3453340
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065494, 0.4065500
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4544455, 0.4542964
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7718404, 0.7718762

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2424

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663893, upper bound: 0.0663676
time: 40.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663892, upper bound: 0.0663658
time: 80.76 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 127.48 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 127.48
Output dim: 5, lower bound: -0.0663603, upper bound: 0.0663958
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 127.48
Output dim: 5, lower bound: -0.0663624, upper bound: 0.0663973
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 127.48
Output dim: 5, lower bound: -0.0663675, upper bound: 0.0663872
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 127.48
Output dim: 5, lower bound: -0.0663659, upper bound: 0.0663826
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 127.48
Output dim: 5, lower bound: -0.0663817, upper bound: 0.0663772
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 127.48
Output dim: 5, lower bound: -0.0663826, upper bound: 0.0663783
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 127.48
Output dim: 5, lower bound: -0.0663893, upper bound: 0.0663676
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 127.48
Output dim: 5, lower bound: -0.0663892, upper bound: 0.0663658

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7088024, 0.7088168
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7433438, 0.7434261
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4773458, 0.4773598
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1816529, 0.1816430
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643795, 0.1643866
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041188, 0.1041163
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3453337, 0.3452036
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065493, 0.4065489
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4542962, 0.4544448
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7718742, 0.7718382

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3054

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0662795, upper bound: 0.0663711
time: 8.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663347, upper bound: 0.0663042
time: 88.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7088023, 0.7088168
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7433434, 0.7434266
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4773456, 0.4773600
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1816529, 0.1816430
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643795, 0.1643865
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041188, 0.1041163
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3453336, 0.3452037
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065495, 0.4065487
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4542957, 0.4544454
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7718741, 0.7718382

Time for backsubstitution: 6.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3054

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0662830, upper bound: 0.0663714
time: 6.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663300, upper bound: 0.0663192
time: 50.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7088786, 0.7087435
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7435367, 0.7432371
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4773502, 0.4773555
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1816519, 0.1816440
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643810, 0.1643850
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041178, 0.1041173
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3453184, 0.3452199
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065488, 0.4065495
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4543797, 0.4543615
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7718657, 0.7718466

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3054

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0662910, upper bound: 0.0663616
time: 84.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663411, upper bound: 0.0663077
time: 110.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7088785, 0.7087436
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7435362, 0.7432376
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4773501, 0.4773556
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1816519, 0.1816440
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643811, 0.1643850
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041177, 0.1041173
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3453183, 0.3452199
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065489, 0.4065493
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4543792, 0.4543620
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7718656, 0.7718467

Time for backsubstitution: 6.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3054

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0662911, upper bound: 0.0663618
time: 7.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663387, upper bound: 0.0663069
time: 15.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7087436, 0.7088786
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7432376, 0.7435362
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4773556, 0.4773501
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1816440, 0.1816519
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643850, 0.1643811
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041173, 0.1041177
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3452199, 0.3453183
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065493, 0.4065489
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4543620, 0.4543792
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7718468, 0.7718657

Time for backsubstitution: 6.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3054

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663009, upper bound: 0.0663501
time: 11.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663521, upper bound: 0.0663002
time: 7.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7087435, 0.7088786
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7432371, 0.7435367
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4773555, 0.4773501
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1816440, 0.1816519
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643850, 0.1643811
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041173, 0.1041178
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3452199, 0.3453184
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065494, 0.4065488
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4543615, 0.4543797
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7718467, 0.7718657

Time for backsubstitution: 6.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3054

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0662987, upper bound: 0.0663481
time: 79.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663521, upper bound: 0.0662917
time: 24.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7088168, 0.7088023
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7434266, 0.7433434
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4773600, 0.4773456
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1816430, 0.1816529
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643865, 0.1643795
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041163, 0.1041188
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3452038, 0.3453336
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065487, 0.4065495
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4544454, 0.4542957
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7718383, 0.7718740

Time for backsubstitution: 6.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3054

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663112, upper bound: 0.0663402
time: 20.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663616, upper bound: 0.0662889
time: 18.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7088167, 0.7088025
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7434261, 0.7433438
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4773598, 0.4773457
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1816430, 0.1816529
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643865, 0.1643795
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041163, 0.1041188
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3452037, 0.3453337
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065489, 0.4065493
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4544448, 0.4542963
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7718382, 0.7718741

Time for backsubstitution: 6.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3054
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 3054

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663129, upper bound: 0.0662866
time: 67.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663567, upper bound: 0.0662872
time: 106.51 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 180.91 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 180.91
Output dim: 5, lower bound: -0.0662795, upper bound: 0.0663711
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 180.91
Output dim: 5, lower bound: -0.0663347, upper bound: 0.0663042
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 180.91
Output dim: 5, lower bound: -0.0662830, upper bound: 0.0663714
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 180.91
Output dim: 5, lower bound: -0.0663300, upper bound: 0.0663192
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 180.91
Output dim: 5, lower bound: -0.0662910, upper bound: 0.0663616
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 180.91
Output dim: 5, lower bound: -0.0663411, upper bound: 0.0663077
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 180.91
Output dim: 5, lower bound: -0.0662911, upper bound: 0.0663618
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 180.91
Output dim: 5, lower bound: -0.0663387, upper bound: 0.0663069
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 180.91
Output dim: 5, lower bound: -0.0663009, upper bound: 0.0663501
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 180.91
Output dim: 5, lower bound: -0.0663521, upper bound: 0.0663002
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 180.91
Output dim: 5, lower bound: -0.0662987, upper bound: 0.0663481
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 180.91
Output dim: 5, lower bound: -0.0663521, upper bound: 0.0662917
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 180.91
Output dim: 5, lower bound: -0.0663112, upper bound: 0.0663402
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 180.91
Output dim: 5, lower bound: -0.0663616, upper bound: 0.0662889
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 180.91
Output dim: 5, lower bound: -0.0663129, upper bound: 0.0662866
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 180.91
Output dim: 5, lower bound: -0.0663567, upper bound: 0.0662872

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7078609, 0.7076808
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7415236, 0.7413387
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4772541, 0.4772502
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1816015, 0.1815991
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643524, 0.1643714
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041122, 0.1041095
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3447018, 0.3447290
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065360, 0.4065355
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4538414, 0.4539345
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7716720, 0.7716107

Time for backsubstitution: 6.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2425

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0662545, upper bound: 0.0663426
time: 13.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0662529, upper bound: 0.0663434
time: 16.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7078608, 0.7076809
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7415231, 0.7413392
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4772539, 0.4772503
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1816015, 0.1815991
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643524, 0.1643714
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041122, 0.1041095
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3447017, 0.3447291
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065361, 0.4065353
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4538409, 0.4539351
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7716718, 0.7716108

Time for backsubstitution: 6.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2425

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0662529, upper bound: 0.0663366
time: 126.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0662532, upper bound: 0.0663458
time: 7.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7079424, 0.7076076
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7417195, 0.7411498
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4772585, 0.4772459
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1816005, 0.1816003
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643539, 0.1643699
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041111, 0.1041110
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3446865, 0.3447452
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065357, 0.4065360
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4539249, 0.4538512
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7716635, 0.7716191

Time for backsubstitution: 6.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2425

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0662657, upper bound: 0.0663267
time: 40.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0662615, upper bound: 0.0663373
time: 5.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7079422, 0.7076077
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7417190, 0.7411503
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4772584, 0.4772459
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1816005, 0.1816003
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643539, 0.1643699
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041111, 0.1041110
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3446864, 0.3447453
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065359, 0.4065359
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4539244, 0.4538517
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7716634, 0.7716193

Time for backsubstitution: 6.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2425

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0662639, upper bound: 0.0663253
time: 95.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0662627, upper bound: 0.0663351
time: 10.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7078022, 0.7077425
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7414173, 0.7414489
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4772638, 0.4772404
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1815926, 0.1816080
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643578, 0.1643659
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041107, 0.1041111
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3445880, 0.3448434
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065360, 0.4065356
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4539072, 0.4538689
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7716445, 0.7716382

Time for backsubstitution: 6.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2425

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0662750, upper bound: 0.0663197
time: 49.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0662719, upper bound: 0.0663209
time: 31.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7076077, 0.7079423
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7411502, 0.7417190
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4772460, 0.4772583
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1816003, 0.1816005
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643699, 0.1643540
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041110, 0.1041111
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3447453, 0.3446864
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065359, 0.4065359
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4538518, 0.4539244
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7716193, 0.7716634

Time for backsubstitution: 6.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2425

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663247, upper bound: 0.0662714
time: 8.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663238, upper bound: 0.0662673
time: 74.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7076075, 0.7079423
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7411497, 0.7417195
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4772458, 0.4772584
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1816003, 0.1816005
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643699, 0.1643539
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041110, 0.1041111
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3447452, 0.3446864
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065360, 0.4065357
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4538512, 0.4539249
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7716192, 0.7716635

Time for backsubstitution: 6.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2425

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663287, upper bound: 0.0662652
time: 21.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663249, upper bound: 0.0662646
time: 16.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7076809, 0.7078608
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7413392, 0.7415233
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4772503, 0.4772539
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1815991, 0.1816015
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643714, 0.1643524
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041095, 0.1041122
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3447291, 0.3447017
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065354, 0.4065362
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4539351, 0.4538409
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7716109, 0.7716718

Time for backsubstitution: 6.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2425

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663357, upper bound: 0.0662597
time: 53.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663347, upper bound: 0.0662657
time: 6.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.4340885, -2.5008628, -3.4340885, -2.5008628, -0.7076808, 0.7078610
1: -4.4212303, -2.8527071, -4.4212303, -2.8527071, -0.7413387, 0.7415237
2: -1.1002389, -0.4291821, -1.1002389, -0.4291821, -0.4772502, 0.4772540
3: -0.5010597, -0.0990435, -0.5010597, -0.0990435, -0.1815991, 0.1816015
4: -1.0829734, -0.5972753, -1.0829734, -0.5972753, -0.1643714, 0.1643524
5: -0.0839282, 0.2280123, -0.0839282, 0.2280123, -0.1041095, 0.1041122
6: -2.1834435, -1.2393150, -2.1834435, -1.2393150, -0.3447290, 0.3447018
7: 0.2227119, 0.7336009, 0.2227119, 0.7336009, -0.4065355, 0.4065360
8: -5.5085545, -4.6044521, -5.5085545, -4.6044521, -0.4539346, 0.4538414
9: -3.6257405, -2.4166441, -3.6257405, -2.4166441, -0.7716107, 0.7716720

Time for backsubstitution: 6.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 401
type: DSZ, layer: 1, pos: 3116
type: DSZ, layer: 1, pos: 2363
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 819
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 386
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3522
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2392
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 3537
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 3067
type: DSZ, layer: 1, pos: 3308
type: DSZ, layer: 1, pos: 3523
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 3521
type: DSZ, layer: 1, pos: 3538
type: DSZ, layer: 1, pos: 598
type: DSZ, layer: 1, pos: 583
type: DSZ, layer: 1, pos: 831
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3508
type: DSZ, layer: 1, pos: 3506
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3492
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 613
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 3493
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3491
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3477
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 567
type: DSZ, layer: 1, pos: 553
type: DSZ, layer: 1, pos: 596
type: DSZ, layer: 1, pos: 3551
type: DSZ, layer: 1, pos: 552
type: DSZ, layer: 1, pos: 581
type: DSZ, layer: 1, pos: 3476
type: DSZ, layer: 1, pos: 3478
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3553
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 760
type: DSZ, layer: 1, pos: 551
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 339
type: DSZ, layer: 1, pos: 3353
type: DSZ, layer: 1, pos: 411
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 578
type: DSZ, layer: 1, pos: 3577
type: DSZ, layer: 1, pos: 636
type: DSZ, layer: 1, pos: 538
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 275
type: DSZ, layer: 1, pos: 276
type: DSZ, layer: 1, pos: 437
type: DSZ, layer: 1, pos: 440
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 447
type: DSZ, layer: 1, pos: 454
type: DSZ, layer: 1, pos: 463
type: DSZ, layer: 1, pos: 482
type: DSZ, layer: 1, pos: 491
type: DSZ, layer: 1, pos: 496
type: DSZ, layer: 1, pos: 497
type: DSZ, layer: 1, pos: 509
type: DSZ, layer: 1, pos: 616
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 633
type: DSZ, layer: 1, pos: 634
type: DSZ, layer: 1, pos: 644
type: DSZ, layer: 1, pos: 649
type: DSZ, layer: 1, pos: 690
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 711
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 730
type: DSZ, layer: 1, pos: 743
type: DSZ, layer: 1, pos: 770
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2303
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2515
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2530
type: DSZ, layer: 1, pos: 2675
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2940
type: DSZ, layer: 1, pos: 2941
type: DSZ, layer: 1, pos: 2942
type: DSZ, layer: 1, pos: 2946
type: DSZ, layer: 1, pos: 2947
type: DSZ, layer: 1, pos: 2964
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3226
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3256
type: DSZ, layer: 1, pos: 3286
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3289
type: DSZ, layer: 1, pos: 3301
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3371
type: DSZ, layer: 1, pos: 3420
type: DSZ, layer: 1, pos: 3421
type: DSZ, layer: 1, pos: 3435
type: DSZ, layer: 1, pos: 3437
type: DSZ, layer: 1, pos: 3542
type: DSZ, layer: 1, pos: 3543
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3557
type: DSZ, layer: 1, pos: 3573
type: DSZ, layer: 1, pos: 3593
type: DSZ, layer: 1, pos: 3594
type: DSZ, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2425

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663336, upper bound: 0.0662598
time: 37.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663326, upper bound: 0.0662630
time: 9.73 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 54.15 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 54.15
Output dim: 5, lower bound: -0.0662545, upper bound: 0.0663426
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 54.15
Output dim: 5, lower bound: -0.0662529, upper bound: 0.0663434
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 54.15
Output dim: 5, lower bound: -0.0662529, upper bound: 0.0663366
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 54.15
Output dim: 5, lower bound: -0.0662532, upper bound: 0.0663458
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 54.15
Output dim: 5, lower bound: -0.0662657, upper bound: 0.0663267
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 54.15
Output dim: 5, lower bound: -0.0662615, upper bound: 0.0663373
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 54.15
Output dim: 5, lower bound: -0.0662639, upper bound: 0.0663253
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 54.15
Output dim: 5, lower bound: -0.0662627, upper bound: 0.0663351
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 54.15
Output dim: 5, lower bound: -0.0662750, upper bound: 0.0663197
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 54.15
Output dim: 5, lower bound: -0.0662719, upper bound: 0.0663209
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 54.15
Output dim: 5, lower bound: -0.0663247, upper bound: 0.0662714
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 54.15
Output dim: 5, lower bound: -0.0663238, upper bound: 0.0662673
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 54.15
Output dim: 5, lower bound: -0.0663287, upper bound: 0.0662652
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 54.15
Output dim: 5, lower bound: -0.0663249, upper bound: 0.0662646
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 54.15
Output dim: 5, lower bound: -0.0663357, upper bound: 0.0662597
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 54.15
Output dim: 5, lower bound: -0.0663347, upper bound: 0.0662657
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 54.15
Output dim: 5, lower bound: -0.0663336, upper bound: 0.0662598
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 54.15
Output dim: 5, lower bound: -0.0663326, upper bound: 0.0662630

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 169.51 + 2057.33 = 2226.84 seconds
