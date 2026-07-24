## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 15)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0363300336


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1716051, 0.1716051)
1: (-3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2153711, 0.2153711)
2: (-2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0304063, 0.0304063)
3: (1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884722, 0.0884722)
4: (-3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0896060, 0.0896060)
5: (1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0727092, 0.0727092)
6: (-2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3776619, 0.3776619)
7: (-2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0868596, 0.0868596)
8: (-0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6698792, 0.6698792)
9: (-3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4177245, 0.4177246)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.14 + 23.13 = 30.27 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0363664, upper bound: 0.0363655

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3367

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3443

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363612, upper bound: 0.0363301
time: 6.59 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0363298, upper bound: 0.0363297
time: 86.48 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 93.15 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 93.15
Output dim: 5, lower bound: -0.0363612, upper bound: 0.0363301
DS_DSZ2, status: Status.VERIFIED, split count: 1, time: 93.15
Output dim: 5, lower bound: -0.0363298, upper bound: 0.0363297

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1715894, 0.1715885
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2152991, 0.2152920
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0302374, 0.0302550
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884391, 0.0884378
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0895568, 0.0895609
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726999, 0.0727031
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3775737, 0.3775818
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0861563, 0.0860832
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6697223, 0.6697042
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4176432, 0.4176439

Time for backsubstitution: 5.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3367

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2518

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363604, upper bound: 0.0363298
time: 134.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363613, upper bound: 0.0363290
time: 4.38 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 144.13 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 144.13
Output dim: 5, lower bound: -0.0363604, upper bound: 0.0363298
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 144.13
Output dim: 5, lower bound: -0.0363613, upper bound: 0.0363290

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1709015, 0.1709237
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2147946, 0.2147686
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0302128, 0.0302307
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884513, 0.0884497
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0893736, 0.0893962
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726906, 0.0726933
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3775542, 0.3775620
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0861177, 0.0860455
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6693845, 0.6693773
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4175417, 0.4175264

Time for backsubstitution: 5.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3367

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2532

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363603, upper bound: 0.0363296
time: 11.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363601, upper bound: 0.0363303
time: 30.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1709245, 0.1709006
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2147757, 0.2147875
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0302131, 0.0302304
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884510, 0.0884500
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0893920, 0.0893778
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726901, 0.0726938
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3775539, 0.3775624
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0861186, 0.0860446
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6693952, 0.6693666
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4175258, 0.4175425

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3367

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2532

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363615, upper bound: 0.0363287
time: 6.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363613, upper bound: 0.0363288
time: 12.58 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 24.27 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 24.27
Output dim: 5, lower bound: -0.0363603, upper bound: 0.0363296
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 24.27
Output dim: 5, lower bound: -0.0363601, upper bound: 0.0363303
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 24.27
Output dim: 5, lower bound: -0.0363615, upper bound: 0.0363287
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 24.27
Output dim: 5, lower bound: -0.0363613, upper bound: 0.0363288

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1700854, 0.1701061
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2141566, 0.2141196
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0301105, 0.0301278
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884385, 0.0884350
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0888065, 0.0888292
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726896, 0.0726923
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3775387, 0.3775468
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0860867, 0.0860145
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6691754, 0.6691654
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4171785, 0.4171607

Time for backsubstitution: 5.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3367

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2387

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363597, upper bound: 0.0363276
time: 5.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363586, upper bound: 0.0363290
time: 74.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1700839, 0.1701075
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2141457, 0.2141311
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0301100, 0.0301284
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884366, 0.0884368
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0888066, 0.0888290
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726897, 0.0726923
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3775389, 0.3775464
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0860867, 0.0860145
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6691728, 0.6691680
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4171761, 0.4171631

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3367

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2387

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363595, upper bound: 0.0363281
time: 4.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363584, upper bound: 0.0363295
time: 11.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1701084, 0.1700830
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2141383, 0.2141384
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0301108, 0.0301275
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884381, 0.0884353
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0888249, 0.0888108
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726891, 0.0726928
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3775384, 0.3775471
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0860876, 0.0860136
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6691861, 0.6691546
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4171624, 0.4171768

Time for backsubstitution: 5.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3367

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2387

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363611, upper bound: 0.0363267
time: 5.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363595, upper bound: 0.0363281
time: 145.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1701069, 0.1700845
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2141268, 0.2141494
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0301103, 0.0301281
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884362, 0.0884372
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0888250, 0.0888107
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726892, 0.0726928
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3775389, 0.3775467
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0860876, 0.0860136
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6691835, 0.6691573
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4171602, 0.4171791

Time for backsubstitution: 5.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3367

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2387

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363610, upper bound: 0.0363272
time: 26.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363593, upper bound: 0.0363283
time: 4.58 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 36.24 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.24
Output dim: 5, lower bound: -0.0363597, upper bound: 0.0363276
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.24
Output dim: 5, lower bound: -0.0363586, upper bound: 0.0363290
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.24
Output dim: 5, lower bound: -0.0363595, upper bound: 0.0363281
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.24
Output dim: 5, lower bound: -0.0363584, upper bound: 0.0363295
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.24
Output dim: 5, lower bound: -0.0363611, upper bound: 0.0363267
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.24
Output dim: 5, lower bound: -0.0363595, upper bound: 0.0363281
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.24
Output dim: 5, lower bound: -0.0363610, upper bound: 0.0363272
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.24
Output dim: 5, lower bound: -0.0363593, upper bound: 0.0363283

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1700839, 0.1701047
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2141548, 0.2141168
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0301100, 0.0301273
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884384, 0.0884349
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0888064, 0.0888291
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726894, 0.0726920
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3775384, 0.3775465
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0860865, 0.0860144
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6691728, 0.6691631
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4171767, 0.4171590

Time for backsubstitution: 5.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3367

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2517

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363589, upper bound: 0.0363258
time: 6.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363585, upper bound: 0.0363292
time: 4.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1700840, 0.1701047
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2141537, 0.2141178
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0301100, 0.0301273
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884384, 0.0884349
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0888064, 0.0888291
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726892, 0.0726921
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3775384, 0.3775465
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0860865, 0.0860143
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6691728, 0.6691631
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4171767, 0.4171588

Time for backsubstitution: 5.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3367

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2517

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363577, upper bound: 0.0363283
time: 6.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363576, upper bound: 0.0363289
time: 13.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1700825, 0.1701062
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2141438, 0.2141282
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0301095, 0.0301279
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884366, 0.0884368
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0888065, 0.0888290
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726894, 0.0726919
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3775389, 0.3775461
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0860865, 0.0860144
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6691704, 0.6691657
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4171743, 0.4171612

Time for backsubstitution: 5.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3367

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2517

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363587, upper bound: 0.0363273
time: 43.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363584, upper bound: 0.0363281
time: 5.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1700825, 0.1701061
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2141427, 0.2141293
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0301095, 0.0301279
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884366, 0.0884368
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0888065, 0.0888290
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726893, 0.0726920
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3775389, 0.3775461
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0860866, 0.0860143
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6691704, 0.6691655
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4171743, 0.4171612

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3367

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2517

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363575, upper bound: 0.0363286
time: 4.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363574, upper bound: 0.0363289
time: 5.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1701070, 0.1700816
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2141364, 0.2141356
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0301103, 0.0301270
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884381, 0.0884353
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0888248, 0.0888107
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726889, 0.0726925
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3775380, 0.3775468
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0860874, 0.0860135
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6691837, 0.6691521
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4171606, 0.4171751

Time for backsubstitution: 5.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3367

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2517

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363608, upper bound: 0.0363262
time: 4.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363594, upper bound: 0.0363265
time: 4.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1701070, 0.1700816
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2141354, 0.2141367
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0301103, 0.0301270
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884381, 0.0884353
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0888248, 0.0888107
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726888, 0.0726926
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3775380, 0.3775468
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0860874, 0.0860134
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6691837, 0.6691521
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4171606, 0.4171751

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3367

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2517

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363591, upper bound: 0.0363269
time: 47.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363583, upper bound: 0.0363277
time: 8.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1701055, 0.1700831
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2141250, 0.2141466
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0301098, 0.0301276
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884362, 0.0884371
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0888249, 0.0888106
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726889, 0.0726924
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3775384, 0.3775464
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0860874, 0.0860135
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6691809, 0.6691550
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4171582, 0.4171774

Time for backsubstitution: 5.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3367

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2517

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363606, upper bound: 0.0363261
time: 22.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363593, upper bound: 0.0363250
time: 88.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1701056, 0.1700831
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2141239, 0.2141476
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0301098, 0.0301276
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884362, 0.0884371
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0888249, 0.0888106
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726888, 0.0726925
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3775384, 0.3775464
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0860875, 0.0860134
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6691809, 0.6691550
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4171584, 0.4171774

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3367

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2517

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363589, upper bound: 0.0363278
time: 25.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363582, upper bound: 0.0363271
time: 18.78 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 49.72 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 49.72
Output dim: 5, lower bound: -0.0363589, upper bound: 0.0363258
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 49.72
Output dim: 5, lower bound: -0.0363585, upper bound: 0.0363292
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 49.72
Output dim: 5, lower bound: -0.0363577, upper bound: 0.0363283
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 49.72
Output dim: 5, lower bound: -0.0363576, upper bound: 0.0363289
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 49.72
Output dim: 5, lower bound: -0.0363587, upper bound: 0.0363273
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 49.72
Output dim: 5, lower bound: -0.0363584, upper bound: 0.0363281
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 49.72
Output dim: 5, lower bound: -0.0363575, upper bound: 0.0363286
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 49.72
Output dim: 5, lower bound: -0.0363574, upper bound: 0.0363289
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 49.72
Output dim: 5, lower bound: -0.0363608, upper bound: 0.0363262
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 49.72
Output dim: 5, lower bound: -0.0363594, upper bound: 0.0363265
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 49.72
Output dim: 5, lower bound: -0.0363591, upper bound: 0.0363269
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 49.72
Output dim: 5, lower bound: -0.0363583, upper bound: 0.0363277
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 49.72
Output dim: 5, lower bound: -0.0363606, upper bound: 0.0363261
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 49.72
Output dim: 5, lower bound: -0.0363593, upper bound: 0.0363250
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 49.72
Output dim: 5, lower bound: -0.0363589, upper bound: 0.0363278
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 49.72
Output dim: 5, lower bound: -0.0363582, upper bound: 0.0363271

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1692910, 0.1693349
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2130057, 0.2129349
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0299964, 0.0300106
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0883984, 0.0883914
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0882405, 0.0882796
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726824, 0.0726846
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3774974, 0.3775049
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0860469, 0.0859757
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6688511, 0.6688532
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4166443, 0.4166070

Time for backsubstitution: 5.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3367

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363494, upper bound: 0.0363270
time: 57.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363591, upper bound: 0.0363175
time: 4.39 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1693141, 0.1693118
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2129729, 0.2129599
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0299933, 0.0300139
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0883949, 0.0883950
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0882581, 0.0882632
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726819, 0.0726849
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3774970, 0.3775053
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0860479, 0.0859749
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6688631, 0.6688412
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4166248, 0.4166262

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3367

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363489, upper bound: 0.0363272
time: 4.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363585, upper bound: 0.0363181
time: 8.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1692911, 0.1693348
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2130047, 0.2129359
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0299964, 0.0300106
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0883984, 0.0883914
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0882405, 0.0882796
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726822, 0.0726847
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3774974, 0.3775049
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0860470, 0.0859756
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6688511, 0.6688532
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4166443, 0.4166069

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3367

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363483, upper bound: 0.0363283
time: 73.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363579, upper bound: 0.0363187
time: 115.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1693141, 0.1693118
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2129718, 0.2129610
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0299933, 0.0300139
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0883949, 0.0883951
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0882581, 0.0882632
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726818, 0.0726851
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3774970, 0.3775053
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0860480, 0.0859748
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6688633, 0.6688412
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4166248, 0.4166262

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3367

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363478, upper bound: 0.0363280
time: 394.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363574, upper bound: 0.0363197
time: 14.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1692895, 0.1693337
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2129832, 0.2129464
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0299958, 0.0300111
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0883965, 0.0883932
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0882406, 0.0882786
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726823, 0.0726845
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3774974, 0.3775046
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0860470, 0.0859757
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6688485, 0.6688546
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4166396, 0.4166093

Time for backsubstitution: 5.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3367

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363492, upper bound: 0.0363272
time: 7.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363589, upper bound: 0.0363166
time: 102.03 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 114.68 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 114.68
Output dim: 5, lower bound: -0.0363494, upper bound: 0.0363270
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 114.68
Output dim: 5, lower bound: -0.0363591, upper bound: 0.0363175
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 114.68
Output dim: 5, lower bound: -0.0363489, upper bound: 0.0363272
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 114.68
Output dim: 5, lower bound: -0.0363585, upper bound: 0.0363181
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 114.68
Output dim: 5, lower bound: -0.0363483, upper bound: 0.0363283
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 114.68
Output dim: 5, lower bound: -0.0363579, upper bound: 0.0363187
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 114.68
Output dim: 5, lower bound: -0.0363478, upper bound: 0.0363280
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 114.68
Output dim: 5, lower bound: -0.0363574, upper bound: 0.0363197
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 114.68
Output dim: 5, lower bound: -0.0363492, upper bound: 0.0363272
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 114.68
Output dim: 5, lower bound: -0.0363589, upper bound: 0.0363166
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 114.68
Output dim: 5, lower bound: -0.0363584, upper bound: 0.0363281
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 114.68
Output dim: 5, lower bound: -0.0363575, upper bound: 0.0363286
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 114.68
Output dim: 5, lower bound: -0.0363574, upper bound: 0.0363289
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 114.68
Output dim: 5, lower bound: -0.0363608, upper bound: 0.0363262
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 114.68
Output dim: 5, lower bound: -0.0363594, upper bound: 0.0363265
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 114.68
Output dim: 5, lower bound: -0.0363591, upper bound: 0.0363269
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 114.68
Output dim: 5, lower bound: -0.0363583, upper bound: 0.0363277
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 114.68
Output dim: 5, lower bound: -0.0363606, upper bound: 0.0363261
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 114.68
Output dim: 5, lower bound: -0.0363593, upper bound: 0.0363250
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 114.68
Output dim: 5, lower bound: -0.0363589, upper bound: 0.0363278
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 114.68
Output dim: 5, lower bound: -0.0363582, upper bound: 0.0363271

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 30.27 + 1770.39 = 1800.66 seconds
