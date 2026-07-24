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
execution time: IAR + RelationalAnalysis = 7.15 + 22.85 = 30.00 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0363664, upper bound: 0.0363655

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 98
type: A, layer: 1, pos: 98
type: B, layer: 1, pos: 3020
type: A, layer: 1, pos: 3020
type: A, layer: 1, pos: 2121
type: B, layer: 1, pos: 2121
type: A, layer: 1, pos: 2199
type: B, layer: 1, pos: 2199
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 97
type: B, layer: 1, pos: 97
type: A, layer: 1, pos: 3443
type: B, layer: 1, pos: 3443
type: B, layer: 1, pos: 3058
type: A, layer: 1, pos: 3058
type: B, layer: 1, pos: 2157
type: A, layer: 1, pos: 2157
type: B, layer: 1, pos: 2110
type: A, layer: 1, pos: 2110
type: A, layer: 1, pos: 789
type: B, layer: 1, pos: 789
type: A, layer: 1, pos: 2592
type: B, layer: 1, pos: 2592
type: A, layer: 1, pos: 806
type: B, layer: 1, pos: 806
type: B, layer: 1, pos: 773
type: A, layer: 1, pos: 773
type: B, layer: 1, pos: 2214
type: A, layer: 1, pos: 2214
type: A, layer: 1, pos: 2209
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 2134
type: A, layer: 1, pos: 2134
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 2135
type: B, layer: 1, pos: 2135
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 2577
type: B, layer: 1, pos: 2577
type: A, layer: 1, pos: 2658
type: B, layer: 1, pos: 2658
type: B, layer: 1, pos: 2448
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 3028
type: B, layer: 1, pos: 3028
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 754
type: B, layer: 1, pos: 754
type: A, layer: 1, pos: 805
type: B, layer: 1, pos: 805
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 860
type: B, layer: 1, pos: 860
type: A, layer: 1, pos: 845
type: B, layer: 1, pos: 845
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 863
type: B, layer: 1, pos: 863
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 2547
type: B, layer: 1, pos: 2547
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 327
type: B, layer: 1, pos: 327
type: A, layer: 1, pos: 3229
type: B, layer: 1, pos: 3229
type: A, layer: 1, pos: 864
type: B, layer: 1, pos: 864
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 3117
type: A, layer: 1, pos: 3117
type: B, layer: 1, pos: 3118
type: A, layer: 1, pos: 3118
type: B, layer: 1, pos: 459
type: A, layer: 1, pos: 459
type: A, layer: 1, pos: 452
type: B, layer: 1, pos: 452
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 2505
type: B, layer: 1, pos: 2505
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 62
type: B, layer: 1, pos: 62
type: A, layer: 1, pos: 458
type: B, layer: 1, pos: 458
type: A, layer: 1, pos: 2517
type: B, layer: 1, pos: 2517
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 823
type: B, layer: 1, pos: 823
type: A, layer: 1, pos: 2065
type: B, layer: 1, pos: 2065
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 2057
type: B, layer: 1, pos: 2057
type: A, layer: 1, pos: 2520
type: B, layer: 1, pos: 2520
type: B, layer: 1, pos: 2486
type: A, layer: 1, pos: 2486
type: B, layer: 1, pos: 2501
type: A, layer: 1, pos: 2501
type: A, layer: 1, pos: 480
type: B, layer: 1, pos: 480
type: A, layer: 1, pos: 705
type: B, layer: 1, pos: 705
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 2066
type: B, layer: 1, pos: 2066
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 2117
type: B, layer: 1, pos: 2117
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2358
type: A, layer: 1, pos: 2358
type: B, layer: 1, pos: 2532
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 678
type: B, layer: 1, pos: 678
type: A, layer: 1, pos: 2133
type: B, layer: 1, pos: 2133
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 2518
type: B, layer: 1, pos: 2518
type: A, layer: 1, pos: 2115
type: B, layer: 1, pos: 2115
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 495
type: A, layer: 1, pos: 716
type: B, layer: 1, pos: 716
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 2484
type: B, layer: 1, pos: 2484
type: A, layer: 1, pos: 2151
type: B, layer: 1, pos: 2151
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 675
type: B, layer: 1, pos: 675
type: A, layer: 1, pos: 2670
type: B, layer: 1, pos: 2670
type: A, layer: 1, pos: 715
type: B, layer: 1, pos: 715
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 2263
type: B, layer: 1, pos: 2263
type: A, layer: 1, pos: 2052
type: B, layer: 1, pos: 2052
type: A, layer: 1, pos: 450
type: B, layer: 1, pos: 450
type: A, layer: 1, pos: 2053
type: B, layer: 1, pos: 2053
type: A, layer: 1, pos: 525
type: B, layer: 1, pos: 525
type: A, layer: 1, pos: 2550
type: B, layer: 1, pos: 2550
type: A, layer: 1, pos: 2373
type: B, layer: 1, pos: 2373
type: A, layer: 1, pos: 3238
type: B, layer: 1, pos: 3238
type: B, layer: 1, pos: 2387
type: A, layer: 1, pos: 2387
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 465
type: A, layer: 1, pos: 465
type: A, layer: 1, pos: 3343
type: B, layer: 1, pos: 3343
type: A, layer: 1, pos: 436
type: A, layer: 1, pos: 441
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 869
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2459
type: A, layer: 1, pos: 2504
type: A, layer: 1, pos: 2669
type: A, layer: 1, pos: 3134
type: A, layer: 1, pos: 3284
type: A, layer: 1, pos: 3314
type: A, layer: 1, pos: 3366
type: A, layer: 1, pos: 3367
type: B, layer: 1, pos: 436
type: B, layer: 1, pos: 441
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 869
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2459
type: B, layer: 1, pos: 2504
type: B, layer: 1, pos: 2669
type: B, layer: 1, pos: 3134
type: B, layer: 1, pos: 3284
type: B, layer: 1, pos: 3314
type: B, layer: 1, pos: 3366
type: B, layer: 1, pos: 3367

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 98

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0363283, upper bound: 0.0363262
time: 284.68 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0363284, upper bound: 0.0363286
time: 4.59 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 289.36 seconds
NS_B1, status: Status.VERIFIED, split count: 1, time: 289.36
Output dim: 5, lower bound: -0.0363283, upper bound: 0.0363262
NS_B2, status: Status.VERIFIED, split count: 1, time: 289.36
Output dim: 5, lower bound: -0.0363284, upper bound: 0.0363286

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 30.00 + 289.36 = 319.36 seconds
