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
execution time: IAR + RelationalAnalysis = 7.95 + 22.23 = 30.17 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0363664, upper bound: 0.0363655

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 741

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2448

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363545, upper bound: 0.0363611
time: 52.80 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363613, upper bound: 0.0363544
time: 6.25 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 59.06 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 59.06
Output dim: 5, lower bound: -0.0363545, upper bound: 0.0363611
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 59.06
Output dim: 5, lower bound: -0.0363613, upper bound: 0.0363544

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1711169, 0.1711036
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2140530, 0.2140185
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0303726, 0.0303772
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884667, 0.0884674
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0885605, 0.0885883
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726650, 0.0726656
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3776581, 0.3776581
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0857043, 0.0857339
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6693819, 0.6693621
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4160665, 0.4160480

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2121

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2387

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363535, upper bound: 0.0363593
time: 4.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363523, upper bound: 0.0363607
time: 4.23 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1711035, 0.1711169
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2140185, 0.2140530
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0303772, 0.0303726
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884674, 0.0884667
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0885883, 0.0885605
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726656, 0.0726650
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3776581, 0.3776582
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0857339, 0.0857043
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6693621, 0.6693819
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4160480, 0.4160665

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2157

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 678

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363611, upper bound: 0.0363549
time: 78.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363617, upper bound: 0.0363538
time: 12.56 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 97.64 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 97.64
Output dim: 5, lower bound: -0.0363535, upper bound: 0.0363593
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 97.64
Output dim: 5, lower bound: -0.0363523, upper bound: 0.0363607
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 97.64
Output dim: 5, lower bound: -0.0363611, upper bound: 0.0363549
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 97.64
Output dim: 5, lower bound: -0.0363617, upper bound: 0.0363538

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1711153, 0.1711020
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2140511, 0.2140157
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0303721, 0.0303767
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884666, 0.0884673
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0885604, 0.0885883
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726648, 0.0726652
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3776578, 0.3776578
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0857040, 0.0857337
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6693795, 0.6693594
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4160644, 0.4160459

Time for backsubstitution: 6.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 897

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363540, upper bound: 0.0363598
time: 4.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363540, upper bound: 0.0363599
time: 5.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1711154, 0.1711020
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2140501, 0.2140167
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0303721, 0.0303767
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884666, 0.0884673
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0885604, 0.0885883
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726646, 0.0726654
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3776578, 0.3776578
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0857041, 0.0857336
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6693797, 0.6693594
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4160644, 0.4160459

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 3343

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2339

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363529, upper bound: 0.0363613
time: 6.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363529, upper bound: 0.0363609
time: 23.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1710277, 0.1710376
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2138463, 0.2138886
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0303604, 0.0303560
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884641, 0.0884637
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0885268, 0.0884970
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726599, 0.0726595
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3776495, 0.3776506
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0857242, 0.0856939
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6693027, 0.6693194
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4160383, 0.4160576

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3117

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3284

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363546, upper bound: 0.0363500
time: 39.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363546, upper bound: 0.0363543
time: 7.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1710242, 0.1710411
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2138543, 0.2138807
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0303606, 0.0303558
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884644, 0.0884635
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0885249, 0.0884990
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726601, 0.0726593
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3776505, 0.3776498
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0857235, 0.0856946
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6692994, 0.6693223
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4160392, 0.4160568

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2373

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 97

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0362975, upper bound: 0.0363405
time: 8.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363481, upper bound: 0.0362895
time: 18.31 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 33.30 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 33.30
Output dim: 5, lower bound: -0.0363540, upper bound: 0.0363598
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.30
Output dim: 5, lower bound: -0.0363540, upper bound: 0.0363599
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 33.30
Output dim: 5, lower bound: -0.0363529, upper bound: 0.0363613
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.30
Output dim: 5, lower bound: -0.0363529, upper bound: 0.0363609
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 33.30
Output dim: 5, lower bound: -0.0363546, upper bound: 0.0363500
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.30
Output dim: 5, lower bound: -0.0363546, upper bound: 0.0363543
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 33.30
Output dim: 5, lower bound: -0.0362975, upper bound: 0.0363405
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.30
Output dim: 5, lower bound: -0.0363481, upper bound: 0.0362895

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1711153, 0.1711020
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2140511, 0.2140157
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0303721, 0.0303767
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884666, 0.0884673
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0885604, 0.0885883
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726648, 0.0726652
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3776578, 0.3776578
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0857040, 0.0857337
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6693795, 0.6693594
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4160644, 0.4160459

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2501

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3443

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363503, upper bound: 0.0363237
time: 5.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363184, upper bound: 0.0363560
time: 5.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1711153, 0.1711020
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2140511, 0.2140157
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0303721, 0.0303767
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884666, 0.0884673
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0885604, 0.0885883
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726648, 0.0726652
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3776578, 0.3776578
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0857040, 0.0857337
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6693795, 0.6693594
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4160644, 0.4160459

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 809

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2135

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363373, upper bound: 0.0363446
time: 32.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363394, upper bound: 0.0363430
time: 5.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1711154, 0.1711020
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2140501, 0.2140167
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0303721, 0.0303767
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884666, 0.0884673
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0885604, 0.0885883
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726646, 0.0726654
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3776578, 0.3776578
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0857041, 0.0857336
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6693797, 0.6693594
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4160644, 0.4160459

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 736

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3058

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363250, upper bound: 0.0363544
time: 24.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363464, upper bound: 0.0363325
time: 4.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1711154, 0.1711020
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2140501, 0.2140167
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0303721, 0.0303767
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884666, 0.0884673
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0885604, 0.0885883
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726646, 0.0726654
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3776578, 0.3776578
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0857041, 0.0857336
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6693797, 0.6693594
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4160644, 0.4160459

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2052

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 129

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0362896, upper bound: 0.0363420
time: 4.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363342, upper bound: 0.0362976
time: 29.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1710277, 0.1710376
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2138463, 0.2138886
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0303604, 0.0303560
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884641, 0.0884637
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0885268, 0.0884970
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726599, 0.0726595
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3776495, 0.3776506
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0857242, 0.0856939
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6693027, 0.6693194
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4160383, 0.4160576

Time for backsubstitution: 6.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2592

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 864

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363472, upper bound: 0.0363407
time: 5.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363469, upper bound: 0.0363413
time: 11.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1710277, 0.1710376
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2138463, 0.2138886
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0303604, 0.0303560
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884641, 0.0884637
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0885268, 0.0884970
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726599, 0.0726595
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3776495, 0.3776506
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0857242, 0.0856939
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6693027, 0.6693194
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4160383, 0.4160576

Time for backsubstitution: 6.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3367

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 511

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363553, upper bound: 0.0363547
time: 147.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363555, upper bound: 0.0363537
time: 22.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1700262, 0.1699010
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2126154, 0.2124730
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0303068, 0.0302857
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884007, 0.0884042
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0884434, 0.0884222
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726382, 0.0726369
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3772848, 0.3773704
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0857166, 0.0856863
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6689162, 0.6688964
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4157466, 0.4156966

Time for backsubstitution: 6.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 3020

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 716

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0362967, upper bound: 0.0363406
time: 6.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0362975, upper bound: 0.0363397
time: 156.12 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1698841, 0.1700431
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2124465, 0.2126418
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0302904, 0.0303020
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884051, 0.0883998
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0884481, 0.0884175
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726377, 0.0726373
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3773711, 0.3772839
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0857151, 0.0856878
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6688735, 0.6689394
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4156791, 0.4157641

Time for backsubstitution: 6.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 458

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2459

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363463, upper bound: 0.0362901
time: 28.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363463, upper bound: 0.0362899
time: 30.75 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 65.81 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 65.81
Output dim: 5, lower bound: -0.0363503, upper bound: 0.0363237
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 65.81
Output dim: 5, lower bound: -0.0363184, upper bound: 0.0363560
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 65.81
Output dim: 5, lower bound: -0.0363373, upper bound: 0.0363446
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 65.81
Output dim: 5, lower bound: -0.0363394, upper bound: 0.0363430
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 65.81
Output dim: 5, lower bound: -0.0363250, upper bound: 0.0363544
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 65.81
Output dim: 5, lower bound: -0.0363464, upper bound: 0.0363325
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 65.81
Output dim: 5, lower bound: -0.0362896, upper bound: 0.0363420
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 65.81
Output dim: 5, lower bound: -0.0363342, upper bound: 0.0362976
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 65.81
Output dim: 5, lower bound: -0.0363472, upper bound: 0.0363407
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 65.81
Output dim: 5, lower bound: -0.0363469, upper bound: 0.0363413
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 65.81
Output dim: 5, lower bound: -0.0363553, upper bound: 0.0363547
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 65.81
Output dim: 5, lower bound: -0.0363555, upper bound: 0.0363537
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 65.81
Output dim: 5, lower bound: -0.0362967, upper bound: 0.0363406
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 65.81
Output dim: 5, lower bound: -0.0362975, upper bound: 0.0363397
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 65.81
Output dim: 5, lower bound: -0.0363463, upper bound: 0.0362901
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 65.81
Output dim: 5, lower bound: -0.0363463, upper bound: 0.0362899

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1710998, 0.1710856
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2139792, 0.2139365
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0302033, 0.0302254
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884335, 0.0884328
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0885112, 0.0885432
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726555, 0.0726591
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3775693, 0.3775774
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0850007, 0.0849573
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6692214, 0.6691835
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4159832, 0.4159654

Time for backsubstitution: 6.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 773

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 678

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363487, upper bound: 0.0363246
time: 6.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363493, upper bound: 0.0363236
time: 89.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1710989, 0.1710865
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2139720, 0.2139437
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0302208, 0.0302078
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884322, 0.0884342
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0885154, 0.0885390
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726586, 0.0726559
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3775774, 0.3775694
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0849277, 0.0850304
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6692033, 0.6692016
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4159839, 0.4159648

Time for backsubstitution: 6.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2339

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 789

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0362632, upper bound: 0.0363508
time: 4.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0363133, upper bound: 0.0362998
time: 38.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1689196, 0.1687998
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2115719, 0.2114196
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0302140, 0.0302162
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0883467, 0.0883474
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0885754, 0.0886034
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0725920, 0.0725925
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3772969, 0.3773142
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0857049, 0.0857346
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6684716, 0.6684077
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4153152, 0.4152658

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2057

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 441

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363247, upper bound: 0.0363326
time: 19.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363247, upper bound: 0.0363446
time: 77.72 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1688132, 0.1689062
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2114550, 0.2115365
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0302117, 0.0302186
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0883467, 0.0883473
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0885756, 0.0886032
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0725920, 0.0725925
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3773143, 0.3772969
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0857050, 0.0857346
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6684277, 0.6684515
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4152843, 0.4152968

Time for backsubstitution: 6.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3443

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3343

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363387, upper bound: 0.0363424
time: 20.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363382, upper bound: 0.0363422
time: 4.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1702223, 0.1701151
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2131841, 0.2130723
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0303303, 0.0303308
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0883905, 0.0883990
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0885578, 0.0885857
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726519, 0.0726537
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3776516, 0.3776522
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0856571, 0.0856905
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6688852, 0.6688163
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4155182, 0.4154584

Time for backsubstitution: 6.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2263

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2592

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363057, upper bound: 0.0363327
time: 5.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363017, upper bound: 0.0363324
time: 6.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1701285, 0.1702088
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2131056, 0.2131507
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0303262, 0.0303349
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0883983, 0.0883912
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0885579, 0.0885857
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726529, 0.0726527
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3776523, 0.3776516
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0856609, 0.0856867
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6688366, 0.6688650
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4154769, 0.4154997

Time for backsubstitution: 6.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 452

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3028

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363372, upper bound: 0.0363248
time: 52.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363389, upper bound: 0.0363230
time: 4.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1709180, 0.1707304
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2138889, 0.2136925
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0303283, 0.0303349
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884474, 0.0884539
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0885603, 0.0885882
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726480, 0.0726477
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3776294, 0.3776358
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0856987, 0.0857276
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6693094, 0.6692131
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4160106, 0.4159337

Time for backsubstitution: 6.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 789

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 864

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0362814, upper bound: 0.0363330
time: 18.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0362812, upper bound: 0.0363336
time: 37.43 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1707438, 0.1709045
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2137259, 0.2138554
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0303303, 0.0303329
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884532, 0.0884481
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0885604, 0.0885881
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726469, 0.0726487
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3776358, 0.3776294
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0856981, 0.0857282
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6692331, 0.6692894
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4159522, 0.4159921

Time for backsubstitution: 6.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 678
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 99

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 863

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0363208, upper bound: 0.0362828
time: 13.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0363185, upper bound: 0.0362854
time: 98.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1704532, 0.1704784
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2125360, 0.2126066
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0303142, 0.0303108
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884600, 0.0884596
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0875519, 0.0875180
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726514, 0.0726509
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3776462, 0.3776468
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0854775, 0.0854462
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6691618, 0.6691749
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4149010, 0.4149491

Time for backsubstitution: 6.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 525

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 897

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363478, upper bound: 0.0363412
time: 14.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363478, upper bound: 0.0363416
time: 5.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1704684, 0.1704631
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2125642, 0.2125784
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0303151, 0.0303098
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884600, 0.0884596
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0875478, 0.0875221
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726513, 0.0726510
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3776460, 0.3776469
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0854765, 0.0854471
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6691580, 0.6691787
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4149296, 0.4149203

Time for backsubstitution: 6.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2115

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2373

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363474, upper bound: 0.0363403
time: 12.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363458, upper bound: 0.0363420
time: 4.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1710231, 0.1710342
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2138507, 0.2138925
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0302522, 0.0302534
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884513, 0.0884513
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0885097, 0.0884848
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726487, 0.0726487
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3776277, 0.3776295
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0857245, 0.0856942
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6692941, 0.6693088
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4160659, 0.4160846

Time for backsubstitution: 6.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 897

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2484

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363549, upper bound: 0.0363538
time: 16.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363549, upper bound: 0.0363546
time: 4.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1710243, 0.1710330
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2138501, 0.2138931
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0302578, 0.0302478
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884518, 0.0884508
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0885146, 0.0884799
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726491, 0.0726483
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3776287, 0.3776287
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0857245, 0.0856942
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6692920, 0.6693109
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4160652, 0.4160851

Time for backsubstitution: 6.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3443

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363515, upper bound: 0.0363181
time: 7.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363187, upper bound: 0.0363492
time: 65.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1699669, 0.1698255
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2124504, 0.2123362
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0302873, 0.0302704
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0883912, 0.0883967
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0883902, 0.0883557
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726363, 0.0726355
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3772705, 0.3773584
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0857104, 0.0856786
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6688752, 0.6688464
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4156852, 0.4156462

Time for backsubstitution: 6.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2339

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 99

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0362449, upper bound: 0.0363289
time: 11.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0362834, upper bound: 0.0362885
time: 10.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1699508, 0.1698416
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2124786, 0.2123080
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0302915, 0.0302662
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0883932, 0.0883947
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0883769, 0.0883690
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726367, 0.0726350
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3772728, 0.3773561
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0857090, 0.0856800
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6688662, 0.6688554
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4156961, 0.4156352

Time for backsubstitution: 6.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 897

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 759

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0362559, upper bound: 0.0363376
time: 5.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0362952, upper bound: 0.0362984
time: 19.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1698841, 0.1700431
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2124465, 0.2126418
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0302904, 0.0303020
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884051, 0.0883998
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0884481, 0.0884175
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726377, 0.0726373
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3773711, 0.3772839
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0857151, 0.0856878
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6688735, 0.6689394
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4156791, 0.4157641

Time for backsubstitution: 6.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 459

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 452

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363476, upper bound: 0.0362830
time: 5.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363407, upper bound: 0.0362895
time: 5.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1698841, 0.1700431
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2124465, 0.2126418
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0302904, 0.0303020
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884051, 0.0883998
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0884481, 0.0884175
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726377, 0.0726373
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3773711, 0.3772839
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0857151, 0.0856878
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6688735, 0.6689394
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4156791, 0.4157641

Time for backsubstitution: 6.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3443
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2387
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2547

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363431, upper bound: 0.0362820
time: 4.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363391, upper bound: 0.0362848
time: 7.49 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 18.62 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0363487, upper bound: 0.0363246
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0363493, upper bound: 0.0363236
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0362632, upper bound: 0.0363508
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0363133, upper bound: 0.0362998
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0363247, upper bound: 0.0363326
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0363247, upper bound: 0.0363446
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0363387, upper bound: 0.0363424
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0363382, upper bound: 0.0363422
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0363057, upper bound: 0.0363327
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0363017, upper bound: 0.0363324
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0363372, upper bound: 0.0363248
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0363389, upper bound: 0.0363230
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0362814, upper bound: 0.0363330
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0362812, upper bound: 0.0363336
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0363208, upper bound: 0.0362828
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0363185, upper bound: 0.0362854
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0363478, upper bound: 0.0363412
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0363478, upper bound: 0.0363416
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0363474, upper bound: 0.0363403
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0363458, upper bound: 0.0363420
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0363549, upper bound: 0.0363538
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0363549, upper bound: 0.0363546
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0363515, upper bound: 0.0363181
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0363187, upper bound: 0.0363492
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0362449, upper bound: 0.0363289
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0362834, upper bound: 0.0362885
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0362559, upper bound: 0.0363376
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0362952, upper bound: 0.0362984
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0363476, upper bound: 0.0362830
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0363407, upper bound: 0.0362895
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0363431, upper bound: 0.0362820
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.62
Output dim: 5, lower bound: -0.0363391, upper bound: 0.0362848

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1710239, 0.1710062
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2138068, 0.2137722
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0301864, 0.0302088
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884303, 0.0884299
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0884497, 0.0884798
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726497, 0.0726536
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3775613, 0.3775702
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0849911, 0.0849470
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6691630, 0.6691220
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4159738, 0.4159570

Time for backsubstitution: 6.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 2133
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 2658

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 520

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363495, upper bound: 0.0363062
time: 14.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363313, upper bound: 0.0363234
time: 5.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2462702, -2.5634642, -3.2462702, -2.5634642, -0.1710205, 0.1710097
1: -3.9141564, -3.1932101, -3.9141564, -3.1932101, -0.2138148, 0.2137642
2: -2.2891426, -2.0782406, -2.2891426, -2.0782406, -0.0301866, 0.0302086
3: 1.2348144, 1.3795028, 1.2348144, 1.3795028, -0.0884305, 0.0884297
4: -3.5540884, -3.1623254, -3.5540884, -3.1623254, -0.0884477, 0.0884817
5: 1.6123576, 1.7493923, 1.6123576, 1.7493923, -0.0726499, 0.0726534
6: -2.1471813, -1.7268512, -2.1471813, -1.7268512, -0.3775623, 0.3775694
7: -2.9855158, -2.6551743, -2.9855158, -2.6551743, -0.0849903, 0.0849477
8: -0.0225523, 0.9098020, -0.0225523, 0.9098020, -0.6691597, 0.6691251
9: -3.9385643, -3.1860199, -3.9385643, -3.1860199, -0.4159746, 0.4159560

Time for backsubstitution: 6.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 716
type: DSZ, layer: 1, pos: 2117
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 465
type: DSZ, layer: 1, pos: 845
type: DSZ, layer: 1, pos: 525
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3314
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2358
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 480
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2504
type: DSZ, layer: 1, pos: 441
type: DSZ, layer: 1, pos: 2214
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2484
type: DSZ, layer: 1, pos: 2157
type: DSZ, layer: 1, pos: 520
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 450
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 705
type: DSZ, layer: 1, pos: 2501
type: DSZ, layer: 1, pos: 805
type: DSZ, layer: 1, pos: 789
type: DSZ, layer: 1, pos: 3229
type: DSZ, layer: 1, pos: 869
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 459
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3118
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 3238
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2057
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2550
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2115
type: DSZ, layer: 1, pos: 327
type: DSZ, layer: 1, pos: 495
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 3284
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2486
type: DSZ, layer: 1, pos: 2592
type: DSZ, layer: 1, pos: 3343
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2459
type: DSZ, layer: 1, pos: 715
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2151
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 3134
type: DSZ, layer: 1, pos: 2670
type: DSZ, layer: 1, pos: 2577
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 452
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2110
type: DSZ, layer: 1, pos: 860
type: DSZ, layer: 1, pos: 458
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 3367
type: DSZ, layer: 1, pos: 806
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3028
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 823
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2135
type: DSZ, layer: 1, pos: 568
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2133

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 129

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0362868, upper bound: 0.0363043
time: 151.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0363315, upper bound: 0.0362605
time: 4.89 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 162.78 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 162.78
Output dim: 5, lower bound: -0.0363495, upper bound: 0.0363062
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 162.78
Output dim: 5, lower bound: -0.0363313, upper bound: 0.0363234
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 162.78
Output dim: 5, lower bound: -0.0362868, upper bound: 0.0363043
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 162.78
Output dim: 5, lower bound: -0.0363315, upper bound: 0.0362605
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 162.78
Output dim: 5, lower bound: -0.0362632, upper bound: 0.0363508
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 162.78
Output dim: 5, lower bound: -0.0363247, upper bound: 0.0363326
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 162.78
Output dim: 5, lower bound: -0.0363247, upper bound: 0.0363446
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 162.78
Output dim: 5, lower bound: -0.0363387, upper bound: 0.0363424
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 162.78
Output dim: 5, lower bound: -0.0363382, upper bound: 0.0363422
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 162.78
Output dim: 5, lower bound: -0.0363057, upper bound: 0.0363327
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 162.78
Output dim: 5, lower bound: -0.0363017, upper bound: 0.0363324
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 162.78
Output dim: 5, lower bound: -0.0363372, upper bound: 0.0363248
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 162.78
Output dim: 5, lower bound: -0.0363389, upper bound: 0.0363230
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 162.78
Output dim: 5, lower bound: -0.0362814, upper bound: 0.0363330
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 162.78
Output dim: 5, lower bound: -0.0362812, upper bound: 0.0363336
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 162.78
Output dim: 5, lower bound: -0.0363478, upper bound: 0.0363412
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 162.78
Output dim: 5, lower bound: -0.0363478, upper bound: 0.0363416
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 162.78
Output dim: 5, lower bound: -0.0363474, upper bound: 0.0363403
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 162.78
Output dim: 5, lower bound: -0.0363458, upper bound: 0.0363420
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 162.78
Output dim: 5, lower bound: -0.0363549, upper bound: 0.0363538
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 162.78
Output dim: 5, lower bound: -0.0363549, upper bound: 0.0363546
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 162.78
Output dim: 5, lower bound: -0.0363515, upper bound: 0.0363181
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 162.78
Output dim: 5, lower bound: -0.0363187, upper bound: 0.0363492
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 162.78
Output dim: 5, lower bound: -0.0362559, upper bound: 0.0363376
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 162.78
Output dim: 5, lower bound: -0.0363476, upper bound: 0.0362830
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 162.78
Output dim: 5, lower bound: -0.0363407, upper bound: 0.0362895
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 162.78
Output dim: 5, lower bound: -0.0363431, upper bound: 0.0362820
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 162.78
Output dim: 5, lower bound: -0.0363391, upper bound: 0.0362848

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 30.17 + 1871.00 = 1901.18 seconds
