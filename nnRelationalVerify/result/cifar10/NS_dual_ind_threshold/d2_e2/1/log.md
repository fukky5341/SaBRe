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
execution time: IAR + RelationalAnalysis = 7.99 + 160.99 = 168.97 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0664757, upper bound: 0.0664833

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 411
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 339
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 3506
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 491
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 3556
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 3229
type: A, layer: 1, pos: 3557
type: A, layer: 1, pos: 3301
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3478
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 3522
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 3256
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3331
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3420
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3537
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 437
type: A, layer: 1, pos: 440
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3371
type: A, layer: 1, pos: 3593
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 411

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664703, upper bound: 0.0661702
time: 55.65 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664693, upper bound: 0.0664792
time: 10.36 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 66.08 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 66.08
Output dim: 5, lower bound: -0.0664703, upper bound: 0.0661702
NS_A2, status: Status.UNKNOWN, split count: 1, time: 66.08
Output dim: 5, lower bound: -0.0664693, upper bound: 0.0664792

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3.4331844, -2.5009477, -3.4334173, -2.5008681, -0.7147098, 0.7150426
1: -4.4190164, -2.8536773, -4.4195561, -2.8527207, -0.7570770, 0.7572021
2: -1.0987442, -0.4309953, -1.0999420, -0.4305194, -0.4747891, 0.4755200
3: -0.4960834, -0.1038258, -0.5009042, -0.1028942, -0.1731185, 0.1771846
4: -1.0823056, -0.5979261, -1.0823295, -0.5974218, -0.1640305, 0.1642159
5: -0.0808873, 0.2276310, -0.0838105, 0.2276923, -0.1006949, 0.1038127
6: -2.1798704, -1.2395329, -2.1832705, -1.2394927, -0.3476076, 0.3504030
7: 0.2241748, 0.7333393, 0.2238538, 0.7332737, -0.4040447, 0.4033329
8: -5.5080533, -4.6052375, -5.5083117, -4.6050673, -0.4564130, 0.4565515
9: -3.6254945, -2.4182944, -3.6253991, -2.4180620, -0.7684146, 0.7698166

Time for backsubstitution: 6.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 339
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 3506
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 491
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 3229
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 3301
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 3478
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 3522
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 3256
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 437
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3593
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3492

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663628, upper bound: 0.0661709
time: 21.16 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664667, upper bound: 0.0661709
time: 28.53 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -3.4340870, -2.5008640, -3.4340878, -2.5008638, -0.7152766, 0.7153673
1: -4.4212294, -2.8527091, -4.4212294, -2.8527081, -0.7579143, 0.7583621
2: -1.1002285, -0.4291834, -1.1002305, -0.4291832, -0.4776633, 0.4764793
3: -0.5010593, -0.0990438, -0.5010595, -0.0990437, -0.1819688, 0.1753179
4: -1.0829659, -0.5972753, -1.0829673, -0.5972753, -0.1649520, 0.1642591
5: -0.0839279, 0.2280119, -0.0839281, 0.2280119, -0.1041614, 0.1036297
6: -2.1834431, -1.2393150, -2.1834435, -1.2393150, -0.3508759, 0.3508631
7: 0.2230258, 0.7336006, 0.2229666, 0.7336007, -0.4060270, 0.4065585
8: -5.5085545, -4.6044631, -5.5085545, -4.6044612, -0.4574419, 0.4572156
9: -3.6257405, -2.4172075, -3.6257405, -2.4170911, -0.7732220, 0.7687807

Time for backsubstitution: 6.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3492
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 339
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 3506
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 491
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 3229
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 3301
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3478
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 3522
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 3256
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 437
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3593
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3492

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663604, upper bound: 0.0664791
time: 32.99 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664656, upper bound: 0.0664775
time: 12.49 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 51.62 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 51.62
Output dim: 5, lower bound: -0.0663628, upper bound: 0.0661709
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 51.62
Output dim: 5, lower bound: -0.0664667, upper bound: 0.0661709
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 51.62
Output dim: 5, lower bound: -0.0663604, upper bound: 0.0664791
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 51.62
Output dim: 5, lower bound: -0.0664656, upper bound: 0.0664775

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -3.4314559, -2.5009573, -3.4313049, -2.5008798, -0.7129993, 0.7129599
1: -4.4189115, -2.8547738, -4.4194260, -2.8540602, -0.7555891, 0.7559590
2: -1.0967977, -0.4313419, -1.0975630, -0.4309447, -0.4726022, 0.4729449
3: -0.4960578, -0.1056322, -0.5008727, -0.1051021, -0.1708863, 0.1753474
4: -1.0808452, -0.5980718, -1.0805485, -0.5976009, -0.1624089, 0.1622992
5: -0.0808599, 0.2268697, -0.0837768, 0.2267624, -0.0997274, 0.1029877
6: -2.1796081, -1.2407594, -2.1829455, -1.2409914, -0.3458804, 0.3489048
7: 0.2275633, 0.7332703, 0.2279943, 0.7331889, -0.4005691, 0.3991329
8: -5.5079212, -4.6062093, -5.5081487, -4.6062541, -0.4550550, 0.4553835
9: -3.6252823, -2.4221876, -3.6251388, -2.4228187, -0.7634624, 0.7657033

Time for backsubstitution: 6.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 339
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 3506
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 491
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 3556
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 3557
type: A, layer: 1, pos: 3229
type: A, layer: 1, pos: 3301
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3478
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 3522
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 3256
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 3331
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3420
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3537
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 437
type: A, layer: 1, pos: 440
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3371
type: A, layer: 1, pos: 3593
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2363

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0662416, upper bound: 0.0661609
time: 14.84 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663564, upper bound: 0.0661577
time: 41.15 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -3.4331615, -2.5009475, -3.4335828, -2.4989533, -0.7170302, 0.7153194
1: -4.4190149, -2.8536787, -4.4208164, -2.8526864, -0.7569623, 0.7595708
2: -1.0987422, -0.4309958, -1.1001976, -0.4273341, -0.4801446, 0.4757207
3: -0.4960832, -0.1038313, -0.5031765, -0.1028847, -0.1731044, 0.1795268
4: -1.0823032, -0.5979263, -1.0824614, -0.5965689, -0.1651073, 0.1640403
5: -0.0808872, 0.2276291, -0.0847630, 0.2278123, -0.1005705, 0.1048822
6: -2.1798701, -1.2395365, -2.1847808, -1.2393341, -0.3476452, 0.3519483
7: 0.2241768, 0.7333393, 0.2237081, 0.7372762, -0.4093560, 0.4034816
8: -5.5080528, -4.6052394, -5.5094624, -4.6048808, -0.4562762, 0.4576074
9: -3.6254909, -2.4182963, -3.6301050, -2.4180493, -0.7683585, 0.7746152

Time for backsubstitution: 6.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 339
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 3506
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 491
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 3556
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 3557
type: A, layer: 1, pos: 3229
type: A, layer: 1, pos: 3301
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3478
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 3522
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 3256
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 3331
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3420
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3537
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 437
type: A, layer: 1, pos: 440
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3371
type: A, layer: 1, pos: 3593
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2363

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663446, upper bound: 0.0661568
time: 38.08 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664595, upper bound: 0.0661585
time: 17.97 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -3.4323590, -2.5008736, -3.4319754, -2.5008755, -0.7135667, 0.7132857
1: -4.4211211, -2.8538046, -4.4210973, -2.8540480, -0.7564266, 0.7571132
2: -1.0982823, -0.4295323, -1.0978515, -0.4296101, -0.4754742, 0.4739014
3: -0.5010335, -0.1008504, -0.5010278, -0.1012515, -0.1797365, 0.1734806
4: -1.0815083, -0.5974227, -1.0811882, -0.5974560, -0.1633334, 0.1623374
5: -0.0839003, 0.2272505, -0.0838941, 0.2270819, -0.1031936, 0.1028045
6: -2.1831758, -1.2405411, -2.1831160, -1.2408134, -0.3491444, 0.3493636
7: 0.2264122, 0.7335294, 0.2271043, 0.7335135, -0.4025507, 0.4023584
8: -5.5084190, -4.6054344, -5.5083885, -4.6056476, -0.4560805, 0.4560450
9: -3.6255260, -2.4211006, -3.6254787, -2.4218497, -0.7682707, 0.7646592

Time for backsubstitution: 6.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 339
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 3506
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 491
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 3556
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 3229
type: A, layer: 1, pos: 3557
type: A, layer: 1, pos: 3301
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3478
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 3522
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 3256
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3331
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3420
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3537
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 437
type: A, layer: 1, pos: 440
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3371
type: A, layer: 1, pos: 3593
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2363

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0662410, upper bound: 0.0664683
time: 6.32 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663517, upper bound: 0.0664647
time: 71.67 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -3.4340634, -2.5008640, -3.4342532, -2.4989491, -0.7175969, 0.7156446
1: -4.4212289, -2.8527095, -4.4224868, -2.8526750, -0.7577999, 0.7607261
2: -1.1002266, -0.4291838, -1.1004857, -0.4259999, -0.4830167, 0.4766802
3: -0.5010593, -0.0990492, -0.5033319, -0.0990342, -0.1819547, 0.1776602
4: -1.0829637, -0.5972753, -1.0830996, -0.5964227, -0.1660295, 0.1640841
5: -0.0839279, 0.2280098, -0.0848805, 0.2281319, -0.1040370, 0.1046993
6: -2.1834433, -1.2393190, -2.1849535, -1.2391579, -0.3509132, 0.3524082
7: 0.2230277, 0.7336004, 0.2228211, 0.7376046, -0.4113391, 0.4067074
8: -5.5085545, -4.6044650, -5.5097046, -4.6042738, -0.4573051, 0.4582725
9: -3.6257367, -2.4172094, -3.6304438, -2.4170787, -0.7731658, 0.7735722

Time for backsubstitution: 6.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2363
type: A, layer: 1, pos: 339
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 3506
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 491
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 3556
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 3229
type: A, layer: 1, pos: 3557
type: A, layer: 1, pos: 3301
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3478
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 3522
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 3256
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3331
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3420
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3537
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 437
type: A, layer: 1, pos: 440
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3371
type: A, layer: 1, pos: 3593
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2363

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663471, upper bound: 0.0664681
time: 45.35 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0664563, upper bound: 0.0664682
time: 29.30 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 81.13 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 81.13
Output dim: 5, lower bound: -0.0662416, upper bound: 0.0661609
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 81.13
Output dim: 5, lower bound: -0.0663564, upper bound: 0.0661577
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 81.13
Output dim: 5, lower bound: -0.0663446, upper bound: 0.0661568
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 81.13
Output dim: 5, lower bound: -0.0664595, upper bound: 0.0661585
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 81.13
Output dim: 5, lower bound: -0.0662410, upper bound: 0.0664683
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 81.13
Output dim: 5, lower bound: -0.0663517, upper bound: 0.0664647
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 81.13
Output dim: 5, lower bound: -0.0663471, upper bound: 0.0664681
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 81.13
Output dim: 5, lower bound: -0.0664563, upper bound: 0.0664682

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3.4314234, -2.5020399, -3.4312785, -2.5017622, -0.7120948, 0.6981114
1: -4.4189110, -2.8563471, -4.4194250, -2.8553424, -0.7542912, 0.7280694
2: -1.0967765, -0.4313472, -1.0975456, -0.4309488, -0.4725567, 0.4728725
3: -0.4958708, -0.1056364, -0.5007199, -0.1051053, -0.1696066, 0.1751833
4: -1.0806630, -0.5980722, -1.0803889, -0.5976012, -0.1617393, 0.1620844
5: -0.0806809, 0.2268692, -0.0836318, 0.2267620, -0.0990273, 0.1028343
6: -2.1790357, -1.2407601, -2.1824782, -1.2409916, -0.3359460, 0.3486407
7: 0.2276158, 0.7332693, 0.2280388, 0.7331880, -0.4005261, 0.3991005
8: -5.5079198, -4.6068602, -5.5081491, -4.6068168, -0.4546801, 0.4486426
9: -3.6252451, -2.4224763, -3.6251082, -2.4230549, -0.7631402, 0.7607853

Time for backsubstitution: 6.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 339
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 3506
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 491
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 3229
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 3301
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3478
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 3522
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 3256
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 437
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3593
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 339

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0662527, upper bound: 0.0657622
time: 9.84 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0662533, upper bound: 0.0660565
time: 9.63 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.4331288, -2.5020304, -3.4335563, -2.4998360, -0.7161251, 0.7004712
1: -4.4190149, -2.8552518, -4.4208155, -2.8539686, -0.7556643, 0.7316814
2: -1.0987209, -0.4310007, -1.1001801, -0.4273378, -0.4800997, 0.4756480
3: -0.4958966, -0.1038355, -0.5030239, -0.1028880, -0.1718248, 0.1793626
4: -1.0821202, -0.5979268, -1.0823016, -0.5965694, -0.1644360, 0.1638255
5: -0.0807083, 0.2276286, -0.0846181, 0.2278119, -0.0998705, 0.1047289
6: -2.1792984, -1.2395371, -2.1843126, -1.2393346, -0.3377082, 0.3516839
7: 0.2242295, 0.7333382, 0.2237526, 0.7372752, -0.4093131, 0.4034489
8: -5.5080528, -4.6058908, -5.5094619, -4.6054435, -0.4559013, 0.4508664
9: -3.6254535, -2.4185853, -3.6300745, -2.4182851, -0.7680364, 0.7696974

Time for backsubstitution: 6.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 339
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 3506
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 491
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 3478
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 3229
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 3301
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 3522
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 3256
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 437
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3593
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 339

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663555, upper bound: 0.0657626
time: 8.31 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663578, upper bound: 0.0660576
time: 9.08 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.4217770, -2.5147197, -3.4316318, -2.5124245, -0.6921033, 0.6994631
1: -4.4058924, -2.8743994, -4.4210958, -2.8713472, -0.7260667, 0.7371758
2: -1.0979722, -0.4296780, -1.0976293, -0.4296389, -0.4750622, 0.4735064
3: -0.4985505, -0.1015595, -0.4989951, -0.1012948, -0.1771083, 0.1706947
4: -1.0802293, -0.5977685, -1.0801235, -0.5974590, -0.1621622, 0.1611713
5: -0.0817238, 0.2270586, -0.0821060, 0.2270769, -0.1007887, 0.1005479
6: -2.1739182, -1.2456671, -2.1754770, -1.2408152, -0.3404411, 0.3378392
7: 0.2264607, 0.7335603, 0.2271367, 0.7335109, -0.4024684, 0.4023193
8: -5.5045891, -4.6110153, -5.5083876, -4.6102500, -0.4477603, 0.4506432
9: -3.6225302, -2.4248428, -3.6253982, -2.4249828, -0.7626802, 0.7610223

Time for backsubstitution: 6.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 339
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 3506
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 491
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 3229
type: B, layer: 1, pos: 3301
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3478
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 3522
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 3256
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 437
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3593
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 339

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0661401, upper bound: 0.0660709
time: 77.61 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0661399, upper bound: 0.0663606
time: 30.10 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.4323266, -2.5019555, -3.4319491, -2.5017579, -0.7126617, 0.6984370
1: -4.4211197, -2.8553779, -4.4210958, -2.8553309, -0.7551289, 0.7292234
2: -1.0982606, -0.4295373, -1.0978343, -0.4296141, -0.4754289, 0.4738284
3: -0.5008459, -0.1008543, -0.5008748, -0.1012548, -0.1784566, 0.1733164
4: -1.0813260, -0.5974231, -1.0810292, -0.5974565, -0.1626636, 0.1621225
5: -0.0837215, 0.2272499, -0.0837474, 0.2270815, -0.1024934, 0.1026511
6: -2.1826015, -1.2405413, -2.1826472, -1.2408139, -0.3392031, 0.3491055
7: 0.2264647, 0.7335284, 0.2271488, 0.7335126, -0.4025079, 0.4023258
8: -5.5084190, -4.6060848, -5.5083885, -4.6062107, -0.4557057, 0.4493038
9: -3.6254890, -2.4213889, -3.6254480, -2.4220846, -0.7679490, 0.7597421

Time for backsubstitution: 6.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 339
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 3506
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 491
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 3229
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 3301
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3478
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 3522
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 3256
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 437
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3593
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 339

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0662523, upper bound: 0.0660664
time: 212.33 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0662538, upper bound: 0.0663630
time: 186.21 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.4234803, -2.5147107, -3.4339087, -2.5104976, -0.6961291, 0.7018196
1: -4.4060011, -2.8733039, -4.4224849, -2.8699732, -0.7274400, 0.7407887
2: -1.0999169, -0.4293299, -1.1002634, -0.4260282, -0.4826053, 0.4762850
3: -0.4985763, -0.0997587, -0.5012990, -0.0990776, -0.1793263, 0.1748744
4: -1.0816822, -0.5976210, -1.0820335, -0.5964261, -0.1648565, 0.1629153
5: -0.0817513, 0.2278179, -0.0830922, 0.2281268, -0.1016320, 0.1024426
6: -2.1741860, -1.2444447, -2.1773143, -1.2391598, -0.3422095, 0.3408840
7: 0.2230764, 0.7336314, 0.2228541, 0.7376021, -0.4112569, 0.4066684
8: -5.5047245, -4.6100450, -5.5097027, -4.6088758, -0.4489853, 0.4528709
9: -3.6227403, -2.4209523, -3.6303642, -2.4202113, -0.7675764, 0.7699353

Time for backsubstitution: 6.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 339
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 3506
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 491
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 3478
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 3229
type: B, layer: 1, pos: 3301
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 3522
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 3256
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 437
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3593
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 339

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0662461, upper bound: 0.0660682
time: 40.47 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0662477, upper bound: 0.0663680
time: 5.44 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.4340308, -2.5019462, -3.4342258, -2.4998312, -0.7166917, 0.7007961
1: -4.4212279, -2.8542824, -4.4224863, -2.8539567, -0.7565020, 0.7328368
2: -1.1002052, -0.4291891, -1.1004679, -0.4260036, -0.4829717, 0.4766072
3: -0.5008716, -0.0990536, -0.5031790, -0.0990376, -0.1806747, 0.1774960
4: -1.0827806, -0.5972758, -1.0829402, -0.5964230, -0.1653584, 0.1638691
5: -0.0837491, 0.2280093, -0.0847338, 0.2281315, -0.1033368, 0.1045459
6: -2.1828687, -1.2393191, -2.1844859, -1.2391584, -0.3409692, 0.3521500
7: 0.2230804, 0.7335994, 0.2228654, 0.7376037, -0.4112963, 0.4066749
8: -5.5085535, -4.6051154, -5.5097036, -4.6048360, -0.4569304, 0.4515311
9: -3.6256993, -2.4174981, -3.6304131, -2.4173141, -0.7728443, 0.7686555

Time for backsubstitution: 6.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 339
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 3506
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 491
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 3478
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 3229
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 3301
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 3522
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 3256
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 437
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3593
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 339

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663560, upper bound: 0.0660736
time: 6.98 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663578, upper bound: 0.0663613
time: 81.10 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 94.46 seconds
NS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 94.46
Output dim: 5, lower bound: -0.0662527, upper bound: 0.0657622
NS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 94.46
Output dim: 5, lower bound: -0.0662533, upper bound: 0.0660565
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 94.46
Output dim: 5, lower bound: -0.0663555, upper bound: 0.0657626
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 94.46
Output dim: 5, lower bound: -0.0663578, upper bound: 0.0660576
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 94.46
Output dim: 5, lower bound: -0.0661401, upper bound: 0.0660709
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 94.46
Output dim: 5, lower bound: -0.0661399, upper bound: 0.0663606
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 94.46
Output dim: 5, lower bound: -0.0662523, upper bound: 0.0660664
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 94.46
Output dim: 5, lower bound: -0.0662538, upper bound: 0.0663630
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 94.46
Output dim: 5, lower bound: -0.0662461, upper bound: 0.0660682
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 94.46
Output dim: 5, lower bound: -0.0662477, upper bound: 0.0663680
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 94.46
Output dim: 5, lower bound: -0.0663560, upper bound: 0.0660736
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 94.46
Output dim: 5, lower bound: -0.0663578, upper bound: 0.0663613

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3.4331162, -2.5026188, -3.4335387, -2.5006351, -0.7151030, 0.6997088
1: -4.4190030, -2.8579674, -4.4208002, -2.8576550, -0.7519823, 0.7289647
2: -1.0971041, -0.4311070, -1.0980269, -0.4274828, -0.4784195, 0.4736559
3: -0.4956202, -0.1038379, -0.5026483, -0.1028913, -0.1715293, 0.1789637
4: -1.0816271, -0.5979477, -1.0816333, -0.5965972, -0.1639202, 0.1631304
5: -0.0802237, 0.2276284, -0.0839592, 0.2278117, -0.0993716, 0.1040521
6: -2.1790752, -1.2396665, -2.1840088, -1.2395105, -0.3373854, 0.3512751
7: 0.2243886, 0.7332350, 0.2239656, 0.7371316, -0.4068340, 0.4013907
8: -5.5079985, -4.6058922, -5.5093870, -4.6054454, -0.4553784, 0.4504330
9: -3.6252851, -2.4191165, -3.6298463, -2.4190063, -0.7668703, 0.7687195

Time for backsubstitution: 6.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 3506
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 491
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 339
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 3556
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 3557
type: A, layer: 1, pos: 3229
type: A, layer: 1, pos: 3301
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3478
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 3522
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 3256
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 3331
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3420
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3537
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 437
type: A, layer: 1, pos: 440
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3371
type: A, layer: 1, pos: 3593
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3491

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663509, upper bound: 0.0656334
time: 9.91 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663508, upper bound: 0.0657542
time: 133.95 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.4330924, -2.5020437, -3.4361095, -2.4998364, -0.7146062, 0.7030129
1: -4.4190145, -2.8553085, -4.4328241, -2.8540344, -0.7530068, 0.7436019
2: -1.0986960, -0.4310037, -1.1004148, -0.4171798, -0.4903186, 0.4747085
3: -0.4958893, -0.1038369, -0.5031561, -0.1023780, -0.1723830, 0.1794086
4: -1.0820723, -0.5979273, -1.0824118, -0.5948061, -0.1662766, 0.1636224
5: -0.0806996, 0.2276283, -0.0846376, 0.2291392, -0.1011958, 0.1043745
6: -2.1792932, -1.2430174, -2.1835845, -1.2429655, -0.3363713, 0.3484727
7: 0.2242347, 0.7333379, 0.2236805, 0.7398847, -0.4067791, 0.4089863
8: -5.5078869, -4.6058931, -5.5102768, -4.6055789, -0.4536752, 0.4511136
9: -3.6254511, -2.4186029, -3.6341562, -2.4183068, -0.7667280, 0.7735627

Time for backsubstitution: 6.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 3506
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 491
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 339
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 3556
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 3557
type: A, layer: 1, pos: 3229
type: A, layer: 1, pos: 3301
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3478
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 3522
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 3256
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 3331
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3420
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3537
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 437
type: A, layer: 1, pos: 440
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3371
type: A, layer: 1, pos: 3593
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3491

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663489, upper bound: 0.0659280
time: 12.58 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663519, upper bound: 0.0660480
time: 20.28 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -3.4217436, -2.5147338, -3.4341855, -2.5124257, -0.6905853, 0.7020074
1: -4.4058919, -2.8744566, -4.4331026, -2.8714128, -0.7234113, 0.7490991
2: -1.0979474, -0.4296808, -1.0978637, -0.4195048, -0.4852785, 0.4725667
3: -0.4985432, -0.1015609, -0.4991238, -0.1007848, -0.1776510, 0.1707396
4: -1.0801816, -0.5977689, -1.0802362, -0.5956966, -0.1640031, 0.1609750
5: -0.0817156, 0.2270583, -0.0821249, 0.2284044, -0.1021071, 0.1001924
6: -2.1739130, -1.2491431, -2.1747441, -1.2444441, -0.3390570, 0.3363557
7: 0.2264663, 0.7335600, 0.2270651, 0.7360990, -0.3999208, 0.4078680
8: -5.5044317, -4.6110187, -5.5091929, -4.6103859, -0.4455455, 0.4508810
9: -3.6225283, -2.4248600, -3.6294789, -2.4250047, -0.7613778, 0.7648873

Time for backsubstitution: 6.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 3506
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 491
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 339
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 3556
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 3557
type: A, layer: 1, pos: 3229
type: A, layer: 1, pos: 3301
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3478
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 3522
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 3256
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 3331
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3420
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3537
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 437
type: A, layer: 1, pos: 440
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3371
type: A, layer: 1, pos: 3593
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3491

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0661333, upper bound: 0.0662374
time: 7.51 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0661340, upper bound: 0.0663576
time: 36.97 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3.4322896, -2.5019696, -3.4345028, -2.5017581, -0.7111433, 0.7009792
1: -4.4211197, -2.8554351, -4.4331036, -2.8553960, -0.7524719, 0.7411418
2: -1.0982356, -0.4295402, -1.0980691, -0.4194806, -0.4856296, 0.4728890
3: -0.5008386, -0.1008558, -0.5010037, -0.1007445, -0.1790150, 0.1733613
4: -1.0812781, -0.5974236, -1.0811418, -0.5956937, -0.1645048, 0.1619258
5: -0.0837130, 0.2272498, -0.0837657, 0.2284090, -0.1038191, 0.1022964
6: -2.1825962, -1.2440178, -2.1819143, -1.2444425, -0.3378674, 0.3459051
7: 0.2264700, 0.7335279, 0.2270765, 0.7361007, -0.3999592, 0.4078676
8: -5.5082617, -4.6060891, -5.5091949, -4.6063457, -0.4534887, 0.4495410
9: -3.6254873, -2.4214075, -3.6295295, -2.4221063, -0.7666446, 0.7635986

Time for backsubstitution: 6.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 3506
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 491
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 339
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 3556
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 3229
type: A, layer: 1, pos: 3557
type: A, layer: 1, pos: 3301
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3478
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 3522
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 3256
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3331
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3420
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3537
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 437
type: A, layer: 1, pos: 440
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3371
type: A, layer: 1, pos: 3593
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3491

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0662478, upper bound: 0.0662370
time: 9.48 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0662456, upper bound: 0.0663616
time: 6.80 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.4234467, -2.5147243, -3.4364610, -2.5104988, -0.6946107, 0.7043633
1: -4.4060001, -2.8733609, -4.4344945, -2.8700390, -0.7247849, 0.7527122
2: -1.0998920, -0.4293326, -1.1004951, -0.4158711, -0.4928382, 0.4753430
3: -0.4985690, -0.0997600, -0.5014299, -0.0985678, -0.1798688, 0.1749199
4: -1.0816345, -0.5976216, -1.0821445, -0.5946630, -0.1666974, 0.1627165
5: -0.0817431, 0.2278177, -0.0831116, 0.2294543, -0.1029504, 0.1020873
6: -2.1741807, -1.2479213, -2.1765840, -1.2427934, -0.3408256, 0.3394007
7: 0.2230818, 0.7336312, 0.2227818, 0.7402219, -0.4087415, 0.4122133
8: -5.5045652, -4.6100492, -5.5105205, -4.6090117, -0.4467683, 0.4531241
9: -3.6227381, -2.4209695, -3.6344461, -2.4202342, -0.7662681, 0.7738010

Time for backsubstitution: 6.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 3506
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 491
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 339
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 3556
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 3557
type: A, layer: 1, pos: 3229
type: A, layer: 1, pos: 3301
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3478
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 3522
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 3256
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 3331
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3420
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3537
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 437
type: A, layer: 1, pos: 440
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3371
type: A, layer: 1, pos: 3593
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3491

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0662392, upper bound: 0.0662363
time: 51.81 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0662415, upper bound: 0.0662344
time: 281.46 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3.4340181, -2.5025344, -3.4342089, -2.5006299, -0.7156694, 0.7000343
1: -4.4212155, -2.8569989, -4.4224691, -2.8576436, -0.7528198, 0.7301193
2: -1.0985879, -0.4292954, -1.0983157, -0.4261487, -0.4812900, 0.4746178
3: -0.5005947, -0.0990558, -0.5028034, -0.0990409, -0.1803790, 0.1770972
4: -1.0822883, -0.5972968, -1.0822725, -0.5964511, -0.1648436, 0.1631736
5: -0.0832635, 0.2280092, -0.0840750, 0.2281313, -0.1028379, 0.1038691
6: -2.1826434, -1.2394489, -2.1841798, -1.2393348, -0.3406447, 0.3517402
7: 0.2232390, 0.7334923, 0.2230777, 0.7374557, -0.4088180, 0.4046177
8: -5.5084982, -4.6051178, -5.5096273, -4.6048393, -0.4564051, 0.4510954
9: -3.6255317, -2.4180298, -3.6301873, -2.4180346, -0.7716805, 0.7676756

Time for backsubstitution: 6.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 3506
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 491
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 339
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 3556
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 3229
type: A, layer: 1, pos: 3557
type: A, layer: 1, pos: 3301
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3478
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 3522
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 3256
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3331
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3420
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3537
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 437
type: A, layer: 1, pos: 440
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3371
type: A, layer: 1, pos: 3593
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3491

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663529, upper bound: 0.0659418
time: 13.54 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663497, upper bound: 0.0660648
time: 7.36 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.4339943, -2.5019598, -3.4367797, -2.4998322, -0.7151725, 0.7033378
1: -4.4212275, -2.8543389, -4.4344950, -2.8540223, -0.7538447, 0.7447555
2: -1.1001803, -0.4291919, -1.1007005, -0.4158468, -0.4931891, 0.4756656
3: -0.5008644, -0.0990550, -0.5033100, -0.0985275, -0.1812330, 0.1775414
4: -1.0827324, -0.5972765, -1.0830508, -0.5946599, -0.1671996, 0.1636699
5: -0.0837405, 0.2280091, -0.0847524, 0.2294589, -0.1046624, 0.1041914
6: -2.1828637, -1.2427961, -2.1837554, -1.2427919, -0.3396328, 0.3489511
7: 0.2230856, 0.7335991, 0.2227933, 0.7402238, -0.4087800, 0.4122128
8: -5.5083947, -4.6051192, -5.5105219, -4.6049724, -0.4547109, 0.4517839
9: -3.6256974, -2.4175158, -3.6344955, -2.4173360, -0.7715338, 0.7725126

Time for backsubstitution: 6.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3491
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 3506
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 491
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 339
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 3556
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 3229
type: A, layer: 1, pos: 3557
type: A, layer: 1, pos: 3301
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3478
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 3522
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 3256
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3331
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3420
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3537
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 437
type: A, layer: 1, pos: 440
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3371
type: A, layer: 1, pos: 3593
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3491

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663521, upper bound: 0.0662355
time: 58.74 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663524, upper bound: 0.0663575
time: 252.20 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 317.60 seconds
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 317.60
Output dim: 5, lower bound: -0.0663509, upper bound: 0.0656334
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 317.60
Output dim: 5, lower bound: -0.0663508, upper bound: 0.0657542
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 317.60
Output dim: 5, lower bound: -0.0663489, upper bound: 0.0659280
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 317.60
Output dim: 5, lower bound: -0.0663519, upper bound: 0.0660480
NS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 317.60
Output dim: 5, lower bound: -0.0661333, upper bound: 0.0662374
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 317.60
Output dim: 5, lower bound: -0.0661340, upper bound: 0.0663576
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 317.60
Output dim: 5, lower bound: -0.0662478, upper bound: 0.0662370
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 317.60
Output dim: 5, lower bound: -0.0662456, upper bound: 0.0663616
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 317.60
Output dim: 5, lower bound: -0.0662392, upper bound: 0.0662363
NS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 317.60
Output dim: 5, lower bound: -0.0662415, upper bound: 0.0662344
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 317.60
Output dim: 5, lower bound: -0.0663529, upper bound: 0.0659418
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 317.60
Output dim: 5, lower bound: -0.0663497, upper bound: 0.0660648
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 317.60
Output dim: 5, lower bound: -0.0663521, upper bound: 0.0662355
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 317.60
Output dim: 5, lower bound: -0.0663524, upper bound: 0.0663575

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.4327121, -2.5026219, -3.4332840, -2.5006363, -0.7146471, 0.6994065
1: -4.4189320, -2.8580167, -4.4207530, -2.8576870, -0.7517243, 0.7286384
2: -1.0968618, -0.4311944, -1.0978705, -0.4275385, -0.4778494, 0.4731909
3: -0.4956141, -0.1043917, -0.5026444, -0.1032509, -0.1711499, 0.1783830
4: -1.0810750, -0.5979826, -1.0812767, -0.5966197, -0.1633525, 0.1627587
5: -0.0802169, 0.2272729, -0.0839548, 0.2275810, -0.0991247, 0.1036768
6: -2.1790116, -1.2403421, -2.1839676, -1.2399490, -0.3368718, 0.3505318
7: 0.2252391, 0.7332187, 0.2245162, 0.7371207, -0.4059753, 0.4008267
8: -5.5079069, -4.6059146, -5.5093279, -4.6054692, -0.4551104, 0.4501675
9: -3.6252027, -2.4201934, -3.6297925, -2.4197021, -0.7661133, 0.7676033

Time for backsubstitution: 6.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 3506
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 491
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 3478
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 3229
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 3301
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 3522
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 3256
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 437
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3593
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 276

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663418, upper bound: 0.0655249
time: 18.24 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663456, upper bound: 0.0656300
time: 14.91 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.4329188, -2.5001185, -3.4332218, -2.5006354, -0.7150923, 0.7023890
1: -4.4194503, -2.8579407, -4.4204330, -2.8576562, -0.7541666, 0.7282155
2: -1.0973536, -0.4290317, -1.0980194, -0.4275045, -0.4781476, 0.4788507
3: -0.4986544, -0.1038495, -0.5026476, -0.1029118, -0.1746544, 0.1788358
4: -1.0817525, -0.5962582, -1.0816228, -0.5965975, -0.1639217, 0.1648910
5: -0.0821598, 0.2277172, -0.0839585, 0.2277876, -0.1013393, 0.1039695
6: -2.1827264, -1.2397356, -2.1840076, -1.2396729, -0.3411249, 0.3509318
7: 0.2242426, 0.7374709, 0.2239677, 0.7371314, -0.4071272, 0.4068616
8: -5.5079532, -4.6055522, -5.5091496, -4.6054473, -0.4557393, 0.4501652
9: -3.6306396, -2.4191101, -3.6298401, -2.4190087, -0.7723058, 0.7687197

Time for backsubstitution: 6.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 3506
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 491
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 3478
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 3229
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 3301
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 3522
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 3256
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 437
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3593
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 276

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663389, upper bound: 0.0656482
time: 25.11 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663461, upper bound: 0.0657496
time: 41.60 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.4326887, -2.5020466, -3.4358549, -2.4998381, -0.7141501, 0.7027102
1: -4.4189425, -2.8553567, -4.4327788, -2.8540661, -0.7527491, 0.7432759
2: -1.0984540, -0.4310908, -1.1002585, -0.4172342, -0.4897492, 0.4742439
3: -0.4958830, -0.1043906, -0.5031518, -0.1027377, -0.1720036, 0.1788276
4: -1.0815198, -0.5979626, -1.0820549, -0.5948287, -0.1657087, 0.1632509
5: -0.0806928, 0.2272728, -0.0846331, 0.2289085, -0.1009488, 0.1039990
6: -2.1792293, -1.2436929, -2.1835423, -1.2434037, -0.3358577, 0.3477292
7: 0.2250856, 0.7333218, 0.2242316, 0.7398731, -0.4059184, 0.4084225
8: -5.5077953, -4.6059170, -5.5102177, -4.6056023, -0.4534078, 0.4508479
9: -3.6253676, -2.4196796, -3.6341021, -2.4190021, -0.7659719, 0.7724465

Time for backsubstitution: 6.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 3506
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 491
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 3478
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 3229
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 3301
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 3522
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 3256
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 437
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3593
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 276

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663409, upper bound: 0.0658217
time: 7.36 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663487, upper bound: 0.0659253
time: 9.02 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.4328949, -2.4995437, -3.4357927, -2.4998376, -0.7145950, 0.7056927
1: -4.4194627, -2.8552816, -4.4324589, -2.8540356, -0.7551916, 0.7428522
2: -1.0989459, -0.4289246, -1.1004072, -0.4171906, -0.4900403, 0.4799040
3: -0.4989234, -0.1038483, -0.5031553, -0.1023985, -0.1755079, 0.1792807
4: -1.0821972, -0.5962380, -1.0824008, -0.5948066, -0.1662779, 0.1653830
5: -0.0826356, 0.2277171, -0.0846369, 0.2291151, -0.1031633, 0.1042919
6: -2.1829443, -1.2430835, -2.1835828, -1.2431276, -0.3401104, 0.3481299
7: 0.2240883, 0.7375759, 0.2236828, 0.7398845, -0.4070735, 0.4144565
8: -5.5078487, -4.6055541, -5.5100408, -4.6055808, -0.4540430, 0.4508452
9: -3.6308050, -2.4185972, -3.6341507, -2.4183097, -0.7721654, 0.7735629

Time for backsubstitution: 6.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 3506
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 491
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 3478
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 3229
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 3301
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 3522
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 3256
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 437
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3593
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 276

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663368, upper bound: 0.0659392
time: 111.78 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663475, upper bound: 0.0660473
time: 11.69 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.4215567, -2.5122366, -3.4338584, -2.5124264, -0.6906027, 0.7046733
1: -4.4063301, -2.8744278, -4.4327369, -2.8714137, -0.7255934, 0.7483621
2: -1.0981992, -0.4276152, -1.0978537, -0.4195141, -0.4850538, 0.4777552
3: -0.5015770, -0.1015728, -0.4991227, -0.1008068, -0.1807731, 0.1706100
4: -1.0803139, -0.5960801, -1.0802250, -0.5956970, -0.1640316, 0.1627351
5: -0.0836512, 0.2271557, -0.0821240, 0.2283806, -0.1040740, 0.1001353
6: -2.1775630, -1.2492096, -2.1747422, -1.2446105, -0.3427936, 0.3360099
7: 0.2263203, 0.7377974, 0.2270674, 0.7360986, -0.4002395, 0.4133323
8: -5.5043864, -4.6106677, -5.5089588, -4.6103878, -0.4459214, 0.4506266
9: -3.6278811, -2.4248538, -3.6294732, -2.4250083, -0.7668163, 0.7648878

Time for backsubstitution: 6.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 3506
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 491
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 3229
type: B, layer: 1, pos: 3301
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3478
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 3522
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 3256
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 437
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3593
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 276

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0661205, upper bound: 0.0662497
time: 122.95 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0661314, upper bound: 0.0663578
time: 7.89 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.4321005, -2.4994693, -3.4341772, -2.5017593, -0.7111592, 0.7036487
1: -4.4215584, -2.8554065, -4.4327393, -2.8553977, -0.7546547, 0.7404056
2: -1.0984879, -0.4274751, -1.0980589, -0.4194899, -0.4854050, 0.4780770
3: -0.5038726, -0.1008681, -0.5010030, -0.1007667, -0.1821370, 0.1732318
4: -1.0814103, -0.5957350, -1.0811307, -0.5956940, -0.1645331, 0.1636855
5: -0.0856486, 0.2273471, -0.0837651, 0.2283851, -0.1057859, 0.1022395
6: -2.1862473, -1.2440840, -2.1819124, -1.2446091, -0.3416055, 0.3455630
7: 0.2263243, 0.7377656, 0.2270791, 0.7361004, -0.4002782, 0.4133320
8: -5.5082192, -4.6057372, -5.5089607, -4.6063480, -0.4538675, 0.4492870
9: -3.6308389, -2.4214003, -3.6295218, -2.4221106, -0.7720823, 0.7635991

Time for backsubstitution: 6.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 3506
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 491
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 3229
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 3301
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3478
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 3522
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 3256
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 437
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3593
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 276

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0662337, upper bound: 0.0662515
time: 15.78 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0662410, upper bound: 0.0661235
time: 204.55 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.4336150, -2.5025377, -3.4339542, -2.5006316, -0.7152140, 0.6997320
1: -4.4211440, -2.8570480, -4.4224234, -2.8576741, -0.7525621, 0.7297924
2: -1.0983461, -0.4293833, -1.0981593, -0.4262046, -0.4807197, 0.4741521
3: -0.5005885, -0.0996097, -0.5027993, -0.0994004, -0.1799996, 0.1765165
4: -1.0817370, -0.5973325, -1.0819163, -0.5964738, -0.1642762, 0.1628008
5: -0.0832565, 0.2276537, -0.0840705, 0.2279005, -0.1025908, 0.1034936
6: -2.1825786, -1.2401247, -2.1841383, -1.2397733, -0.3401298, 0.3509967
7: 0.2240894, 0.7334757, 0.2236281, 0.7374448, -0.4079593, 0.4040529
8: -5.5084052, -4.6051407, -5.5095682, -4.6048622, -0.4561365, 0.4508296
9: -3.6254487, -2.4191065, -3.6301336, -2.4187312, -0.7709234, 0.7665591

Time for backsubstitution: 6.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 3506
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 491
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 3478
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 3229
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 3301
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 3522
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 3256
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 437
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3593
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 276

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663399, upper bound: 0.0658335
time: 11.40 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663471, upper bound: 0.0659399
time: 21.47 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.4338214, -2.5000346, -3.4338923, -2.5006309, -0.7156590, 0.7027143
1: -4.4216609, -2.8569729, -4.4221034, -2.8576448, -0.7550045, 0.7293705
2: -1.0988367, -0.4272218, -1.0983081, -0.4261709, -0.4810179, 0.4798087
3: -0.5036290, -0.0990673, -0.5028026, -0.0990611, -0.1835036, 0.1769694
4: -1.0824146, -0.5956088, -1.0822620, -0.5964514, -0.1648448, 0.1649326
5: -0.0851990, 0.2280979, -0.0840743, 0.2281072, -0.1048050, 0.1037864
6: -2.1862922, -1.2395196, -2.1841784, -1.2394971, -0.3443813, 0.3513962
7: 0.2230915, 0.7377294, 0.2230801, 0.7374554, -0.4091123, 0.4100894
8: -5.5084515, -4.6047783, -5.5093899, -4.6048408, -0.4567661, 0.4508276
9: -3.6308837, -2.4180226, -3.6301813, -2.4180384, -0.7771143, 0.7676760

Time for backsubstitution: 6.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 3506
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 491
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 3478
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 3229
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 3301
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 3522
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 3256
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 437
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3593
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 276

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663385, upper bound: 0.0659575
time: 15.06 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663460, upper bound: 0.0660627
time: 8.17 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.4335911, -2.5019624, -3.4365253, -2.4998331, -0.7147169, 0.7030357
1: -4.4211550, -2.8543890, -4.4344482, -2.8540542, -0.7535869, 0.7444283
2: -1.0999382, -0.4292797, -1.1005443, -0.4159014, -0.4926192, 0.4752004
3: -0.5008583, -0.0996087, -0.5033058, -0.0988872, -0.1808535, 0.1769605
4: -1.0821812, -0.5973118, -1.0826944, -0.5946826, -0.1666323, 0.1632972
5: -0.0837335, 0.2276536, -0.0847480, 0.2292281, -0.1044153, 0.1038160
6: -2.1827989, -1.2434714, -2.1837127, -1.2432301, -0.3391182, 0.3482074
7: 0.2239358, 0.7335821, 0.2233440, 0.7402117, -0.4079197, 0.4116488
8: -5.5083032, -4.6051426, -5.5104618, -4.6049957, -0.4544428, 0.4515179
9: -3.6256137, -2.4185929, -3.6344419, -2.4180322, -0.7707775, 0.7713958

Time for backsubstitution: 6.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 3506
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 491
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 3478
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 3229
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 3301
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 3522
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 3256
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 437
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3593
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 276

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663416, upper bound: 0.0661303
time: 7.49 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663479, upper bound: 0.0662349
time: 6.06 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.4337976, -2.4994593, -3.4364629, -2.4998331, -0.7151623, 0.7060181
1: -4.4216728, -2.8543129, -4.4341283, -2.8540235, -0.7560296, 0.7440059
2: -1.1004287, -0.4271149, -1.1006930, -0.4158579, -0.4929104, 0.4808573
3: -0.5038986, -0.0990666, -0.5033092, -0.0985479, -0.1843574, 0.1774136
4: -1.0828590, -0.5955882, -1.0830400, -0.5946602, -0.1672006, 0.1654288
5: -0.0856759, 0.2280978, -0.0847516, 0.2294348, -0.1066293, 0.1041088
6: -2.1865120, -1.2428634, -2.1837535, -1.2429540, -0.3433690, 0.3486077
7: 0.2229376, 0.7378379, 0.2227955, 0.7402234, -0.4090757, 0.4176843
8: -5.5083547, -4.6047792, -5.5102863, -4.6049738, -0.4550786, 0.4515154
9: -3.6310487, -2.4175105, -3.6344900, -2.4173388, -0.7769699, 0.7725130

Time for backsubstitution: 6.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 276
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 3506
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 491
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 3478
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 3229
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 3301
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 3522
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 3256
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 437
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3593
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 276

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663357, upper bound: 0.0662514
time: 19.86 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663468, upper bound: 0.0663558
time: 16.12 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 42.59 seconds
NS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 42.59
Output dim: 5, lower bound: -0.0663418, upper bound: 0.0655249
NS_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 42.59
Output dim: 5, lower bound: -0.0663456, upper bound: 0.0656300
NS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 42.59
Output dim: 5, lower bound: -0.0663389, upper bound: 0.0656482
NS_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 42.59
Output dim: 5, lower bound: -0.0663461, upper bound: 0.0657496
NS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 42.59
Output dim: 5, lower bound: -0.0663409, upper bound: 0.0658217
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 42.59
Output dim: 5, lower bound: -0.0663487, upper bound: 0.0659253
NS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 42.59
Output dim: 5, lower bound: -0.0663368, upper bound: 0.0659392
NS_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 42.59
Output dim: 5, lower bound: -0.0663475, upper bound: 0.0660473
NS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 42.59
Output dim: 5, lower bound: -0.0661205, upper bound: 0.0662497
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 42.59
Output dim: 5, lower bound: -0.0661314, upper bound: 0.0663578
NS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 42.59
Output dim: 5, lower bound: -0.0662337, upper bound: 0.0662515
NS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 42.59
Output dim: 5, lower bound: -0.0662410, upper bound: 0.0661235
NS_A2_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 42.59
Output dim: 5, lower bound: -0.0663399, upper bound: 0.0658335
NS_A2_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 42.59
Output dim: 5, lower bound: -0.0663471, upper bound: 0.0659399
NS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 42.59
Output dim: 5, lower bound: -0.0663385, upper bound: 0.0659575
NS_A2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 42.59
Output dim: 5, lower bound: -0.0663460, upper bound: 0.0660627
NS_A2_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 42.59
Output dim: 5, lower bound: -0.0663416, upper bound: 0.0661303
NS_A2_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 42.59
Output dim: 5, lower bound: -0.0663479, upper bound: 0.0662349
NS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 42.59
Output dim: 5, lower bound: -0.0663357, upper bound: 0.0662514
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 42.59
Output dim: 5, lower bound: -0.0663468, upper bound: 0.0663558

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.4326699, -2.5021307, -3.4429572, -2.4999342, -0.7128642, 0.7104028
1: -4.4169407, -2.8553581, -4.4305253, -2.8547959, -0.7530216, 0.7422014
2: -1.0984322, -0.4311763, -1.1036806, -0.4171208, -0.4891194, 0.4767120
3: -0.4956950, -0.1043928, -0.5030038, -0.1025662, -0.1722860, 0.1784730
4: -1.0815108, -0.5980495, -1.0834537, -0.5946041, -0.1654376, 0.1643603
5: -0.0805934, 0.2272720, -0.0845729, 0.2292450, -0.1013745, 0.1036603
6: -2.1790009, -1.2437352, -2.1833220, -1.2375712, -0.3404764, 0.3446189
7: 0.2256034, 0.7332489, 0.2248367, 0.7406228, -0.4066959, 0.4054734
8: -5.5077481, -4.6059351, -5.5153627, -4.6050172, -0.4522520, 0.4557757
9: -3.6253600, -2.4197035, -3.6357846, -2.4190044, -0.7652916, 0.7746546

Time for backsubstitution: 6.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 3506
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 491
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 339
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 3556
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 3557
type: A, layer: 1, pos: 3229
type: A, layer: 1, pos: 3301
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3478
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 3522
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 3256
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 3331
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3420
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3537
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 437
type: A, layer: 1, pos: 440
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3371
type: A, layer: 1, pos: 3593
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3493

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663478, upper bound: 0.0657444
time: 187.11 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663484, upper bound: 0.0659242
time: 40.00 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.4215376, -2.5123210, -3.4409726, -2.5125222, -0.6893151, 0.7123809
1: -4.4043298, -2.8744283, -4.4304829, -2.8721452, -0.7258676, 0.7472857
2: -1.0981779, -0.4276995, -1.1012772, -0.4194022, -0.4844224, 0.4802245
3: -0.5013883, -0.1015753, -0.4989753, -0.1006358, -0.1810563, 0.1702556
4: -1.0803045, -0.5961672, -1.0816219, -0.5954767, -0.1637662, 0.1638414
5: -0.0835516, 0.2271550, -0.0820615, 0.2287170, -0.1044996, 0.0997962
6: -2.1773341, -1.2492511, -2.1745205, -1.2387795, -0.3474026, 0.3329026
7: 0.2268378, 0.7377256, 0.2276722, 0.7368535, -0.4010173, 0.4103833
8: -5.5043402, -4.6106863, -5.5141058, -4.6098032, -0.4447667, 0.4555580
9: -3.6278732, -2.4248781, -3.6311543, -2.4250100, -0.7661368, 0.7670961

Time for backsubstitution: 6.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 3506
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 491
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 339
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 3556
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 3478
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 3557
type: A, layer: 1, pos: 3229
type: A, layer: 1, pos: 3301
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 3522
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 3256
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 3331
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3420
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3537
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 437
type: A, layer: 1, pos: 440
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3371
type: A, layer: 1, pos: 3593
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3493

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0661321, upper bound: 0.0658856
time: 75.56 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0661326, upper bound: 0.0663574
time: 5.47 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.4337783, -2.4995432, -3.4435706, -2.4999290, -0.7138761, 0.7137223
1: -4.4196711, -2.8543139, -4.4318743, -2.8547549, -0.7563033, 0.7429290
2: -1.1004075, -0.4271995, -1.1041160, -0.4157437, -0.4922801, 0.4833251
3: -0.5037091, -0.0990688, -0.5031611, -0.0983764, -0.1846397, 0.1770598
4: -1.0828496, -0.5956755, -1.0844381, -0.5944388, -0.1669353, 0.1665374
5: -0.0855764, 0.2280971, -0.0846902, 0.2297713, -0.1070549, 0.1037697
6: -2.1862833, -1.2429050, -2.1835313, -1.2371248, -0.3479877, 0.3454980
7: 0.2234554, 0.7377655, 0.2234006, 0.7409739, -0.4098505, 0.4147348
8: -5.5083075, -4.6047974, -5.5154262, -4.6043906, -0.4539229, 0.4564421
9: -3.6310401, -2.4175346, -3.6361709, -2.4173398, -0.7762901, 0.7747222

Time for backsubstitution: 6.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3493
type: A, layer: 1, pos: 3506
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 551
type: A, layer: 1, pos: 3054
type: A, layer: 1, pos: 386
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 491
type: A, layer: 1, pos: 3476
type: A, layer: 1, pos: 2576
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 3067
type: A, layer: 1, pos: 482
type: A, layer: 1, pos: 567
type: A, layer: 1, pos: 401
type: A, layer: 1, pos: 581
type: A, layer: 1, pos: 497
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 463
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 2392
type: A, layer: 1, pos: 583
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 3521
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 3068
type: A, layer: 1, pos: 3508
type: A, layer: 1, pos: 774
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 275
type: A, layer: 1, pos: 339
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 3492
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 3556
type: A, layer: 1, pos: 276
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 2941
type: A, layer: 1, pos: 3478
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 3229
type: A, layer: 1, pos: 3557
type: A, layer: 1, pos: 3301
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 3553
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 3116
type: A, layer: 1, pos: 760
type: A, layer: 1, pos: 3477
type: A, layer: 1, pos: 2940
type: A, layer: 1, pos: 2488
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 743
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 171
type: A, layer: 1, pos: 2942
type: A, layer: 1, pos: 596
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 3421
type: A, layer: 1, pos: 819
type: A, layer: 1, pos: 553
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 3552
type: A, layer: 1, pos: 690
type: A, layer: 1, pos: 2524
type: A, layer: 1, pos: 3522
type: A, layer: 1, pos: 633
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 3551
type: A, layer: 1, pos: 3115
type: A, layer: 1, pos: 3308
type: A, layer: 1, pos: 3226
type: A, layer: 1, pos: 636
type: A, layer: 1, pos: 598
type: A, layer: 1, pos: 496
type: A, layer: 1, pos: 3326
type: A, layer: 1, pos: 3353
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 578
type: A, layer: 1, pos: 634
type: A, layer: 1, pos: 552
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 2530
type: A, layer: 1, pos: 730
type: A, layer: 1, pos: 2425
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 2211
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 616
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2119
type: A, layer: 1, pos: 2515
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 831
type: A, layer: 1, pos: 3256
type: A, layer: 1, pos: 3577
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 613
type: A, layer: 1, pos: 3538
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 3523
type: A, layer: 1, pos: 3331
type: A, layer: 1, pos: 2947
type: A, layer: 1, pos: 2676
type: A, layer: 1, pos: 3573
type: A, layer: 1, pos: 3420
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2946
type: A, layer: 1, pos: 454
type: A, layer: 1, pos: 718
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 3537
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 3286
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 185
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2964
type: A, layer: 1, pos: 830
type: A, layer: 1, pos: 2675
type: A, layer: 1, pos: 698
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 853
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 621
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 829
type: A, layer: 1, pos: 3435
type: A, layer: 1, pos: 814
type: A, layer: 1, pos: 538
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 437
type: A, layer: 1, pos: 440
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 447
type: A, layer: 1, pos: 509
type: A, layer: 1, pos: 644
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2696
type: A, layer: 1, pos: 3371
type: A, layer: 1, pos: 3593
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3598

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3493

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663492, upper bound: 0.0662865
time: 8.77 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0663488, upper bound: 0.0663522
time: 85.19 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 100.55 seconds
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 100.55
Output dim: 5, lower bound: -0.0663478, upper bound: 0.0657444
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 100.55
Output dim: 5, lower bound: -0.0663484, upper bound: 0.0659242
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 100.55
Output dim: 5, lower bound: -0.0661321, upper bound: 0.0658856
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 100.55
Output dim: 5, lower bound: -0.0661326, upper bound: 0.0663574
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 100.55
Output dim: 5, lower bound: -0.0663492, upper bound: 0.0662865
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 100.55
Output dim: 5, lower bound: -0.0663488, upper bound: 0.0663522

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.4328787, -2.5001049, -3.4429564, -2.4999344, -0.7132566, 0.7127920
1: -4.4180965, -2.8553290, -4.4305248, -2.8548024, -0.7551514, 0.7422155
2: -1.0986712, -0.4283980, -1.1036801, -0.4171211, -0.4893349, 0.4816480
3: -0.4979554, -0.1043757, -0.5030036, -0.1025670, -0.1746287, 0.1784801
4: -1.0816395, -0.5971307, -1.0834508, -0.5946043, -0.1655091, 0.1652533
5: -0.0816126, 0.2273941, -0.0845729, 0.2292447, -0.1023569, 0.1037101
6: -2.1803946, -1.2435997, -2.1833215, -1.2375720, -0.3418921, 0.3446728
7: 0.2254726, 0.7371930, 0.2248382, 0.7406227, -0.4070692, 0.4107556
8: -5.5086465, -4.6057701, -5.5153627, -4.6050239, -0.4531042, 0.4558648
9: -3.6297281, -2.4196901, -3.6357849, -2.4190061, -0.7697061, 0.7746686

Time for backsubstitution: 6.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3506
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 491
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 3478
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 3229
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 3301
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 3522
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 3256
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 437
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3593
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3506

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0662044, upper bound: 0.0659177
time: 14.75 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663452, upper bound: 0.0659216
time: 7.07 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.4217451, -2.5102952, -3.4409721, -2.5125227, -0.6896707, 0.7147785
1: -4.4054756, -2.8744037, -4.4304838, -2.8721514, -0.7279885, 0.7472989
2: -1.0984089, -0.4249243, -1.1012762, -0.4194024, -0.4846328, 0.4851652
3: -0.5036495, -0.1015605, -0.4989752, -0.1006367, -0.1833982, 0.1702617
4: -1.0804285, -0.5952464, -1.0816189, -0.5954769, -0.1638498, 0.1647342
5: -0.0845712, 0.2272862, -0.0820614, 0.2287164, -0.1054818, 0.0998550
6: -2.1787288, -1.2491121, -2.1745203, -1.2387800, -0.3488200, 0.3329560
7: 0.2267109, 0.7416726, 0.2276739, 0.7368537, -0.4013877, 0.4156823
8: -5.5052371, -4.6104965, -5.5141058, -4.6098094, -0.4456192, 0.4556774
9: -3.6322420, -2.4248655, -3.6311538, -2.4250114, -0.7705573, 0.7671084

Time for backsubstitution: 6.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3506
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 491
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 3229
type: B, layer: 1, pos: 3301
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 3478
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 3522
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 3256
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 437
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3593
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3506

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0659896, upper bound: 0.0663546
time: 6.60 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0661279, upper bound: 0.0663514
time: 35.53 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.4329464, -2.4995480, -3.4429352, -2.4999318, -0.7130598, 0.7130958
1: -4.4196219, -2.8547828, -4.4318399, -2.8551159, -0.7559097, 0.7424271
2: -1.0996356, -0.4273559, -1.1035254, -0.4158577, -0.4914355, 0.4826447
3: -0.5036978, -0.0999132, -0.5031524, -0.0990238, -0.1839817, 0.1762050
4: -1.0821519, -0.5957378, -1.0839019, -0.5944865, -0.1661854, 0.1659490
5: -0.0855647, 0.2277167, -0.0846811, 0.2294796, -0.1067331, 0.1033586
6: -2.1861706, -1.2434355, -2.1834440, -1.2375318, -0.3474862, 0.3448915
7: 0.2250319, 0.7377377, 0.2246065, 0.7409513, -0.4082632, 0.4135083
8: -5.5082502, -4.6051641, -5.5153832, -4.6046724, -0.4535690, 0.4560118
9: -3.6309500, -2.4192338, -3.6361024, -2.4186430, -0.7749246, 0.7729679

Time for backsubstitution: 6.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3506
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 551
type: B, layer: 1, pos: 3054
type: B, layer: 1, pos: 386
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 491
type: B, layer: 1, pos: 3476
type: B, layer: 1, pos: 2576
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3067
type: B, layer: 1, pos: 482
type: B, layer: 1, pos: 401
type: B, layer: 1, pos: 567
type: B, layer: 1, pos: 581
type: B, layer: 1, pos: 497
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 463
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 2392
type: B, layer: 1, pos: 583
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 411
type: B, layer: 1, pos: 3521
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 3068
type: B, layer: 1, pos: 3508
type: B, layer: 1, pos: 774
type: B, layer: 1, pos: 3491
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 275
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2363
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 3493
type: B, layer: 1, pos: 3556
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 2941
type: B, layer: 1, pos: 3478
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 3557
type: B, layer: 1, pos: 3229
type: B, layer: 1, pos: 3301
type: B, layer: 1, pos: 3553
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 3116
type: B, layer: 1, pos: 3477
type: B, layer: 1, pos: 760
type: B, layer: 1, pos: 2940
type: B, layer: 1, pos: 2488
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 743
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 171
type: B, layer: 1, pos: 2942
type: B, layer: 1, pos: 596
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 3421
type: B, layer: 1, pos: 819
type: B, layer: 1, pos: 553
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 3552
type: B, layer: 1, pos: 690
type: B, layer: 1, pos: 2524
type: B, layer: 1, pos: 633
type: B, layer: 1, pos: 3522
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 3551
type: B, layer: 1, pos: 3115
type: B, layer: 1, pos: 3308
type: B, layer: 1, pos: 3226
type: B, layer: 1, pos: 636
type: B, layer: 1, pos: 598
type: B, layer: 1, pos: 496
type: B, layer: 1, pos: 3326
type: B, layer: 1, pos: 3353
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 578
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 634
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 552
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 2530
type: B, layer: 1, pos: 730
type: B, layer: 1, pos: 2425
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 616
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2119
type: B, layer: 1, pos: 2515
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 831
type: B, layer: 1, pos: 3256
type: B, layer: 1, pos: 3577
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 613
type: B, layer: 1, pos: 3538
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 3523
type: B, layer: 1, pos: 2947
type: B, layer: 1, pos: 3331
type: B, layer: 1, pos: 2676
type: B, layer: 1, pos: 3573
type: B, layer: 1, pos: 3420
type: B, layer: 1, pos: 2946
type: B, layer: 1, pos: 695
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 454
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 3537
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 3286
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2964
type: B, layer: 1, pos: 185
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 830
type: B, layer: 1, pos: 2675
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 853
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 621
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 829
type: B, layer: 1, pos: 3435
type: B, layer: 1, pos: 814
type: B, layer: 1, pos: 538
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 437
type: B, layer: 1, pos: 440
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 447
type: B, layer: 1, pos: 509
type: B, layer: 1, pos: 644
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2696
type: B, layer: 1, pos: 3371
type: B, layer: 1, pos: 3593
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3598

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3506

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0662090, upper bound: 0.0662830
time: 9.46 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0663425, upper bound: 0.0662787
time: 75.43 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 168.97 + 3482.01 = 3650.99 seconds
