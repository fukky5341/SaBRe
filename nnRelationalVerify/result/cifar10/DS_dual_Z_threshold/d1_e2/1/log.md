## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 1)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.026144530399999998


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.2363062, -2.5159841, -3.2363062, -2.5159841, -0.2151082, 0.2151082)
1: (-1.2228723, -0.3951268, -1.2228723, -0.3951268, -0.2485532, 0.2485532)
2: (-1.9681013, -1.6588017, -1.9681013, -1.6588017, -0.0567758, 0.0567758)
3: (-0.7469844, -0.4570597, -0.7469844, -0.4570597, -0.0747739, 0.0747739)
4: (-2.7783232, -2.2692385, -2.7783232, -2.2692385, -0.1083375, 0.1083375)
5: (-0.2394420, 0.0989661, -0.2394420, 0.0989661, -0.0720247, 0.0720247)
6: (-0.1065651, 0.2169087, -0.1065651, 0.2169087, -0.2161175, 0.2161176)
7: (-0.7139419, -0.3450459, -0.7139419, -0.3450459, -0.0640448, 0.0640448)
8: (-2.5324945, -1.6453166, -2.5324945, -1.6453166, -0.6475120, 0.6475117)
9: (-1.5339894, -0.6906605, -1.5339894, -0.6906605, -0.4303203, 0.4303203)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 8.25 + 194.22 = 202.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0262224, upper bound: 0.0262229

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2829
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 259
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 3541

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2604

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261977, upper bound: 0.0261788
time: 72.01 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261775, upper bound: 0.0261986
time: 34.12 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 106.20 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 106.20
Output dim: 3, lower bound: -0.0261977, upper bound: 0.0261788
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 106.20
Output dim: 3, lower bound: -0.0261775, upper bound: 0.0261986

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.2363062, -2.5159841, -3.2363062, -2.5159841, -0.2115353, 0.2116242
1: -1.2228723, -0.3951268, -1.2228723, -0.3951268, -0.2452658, 0.2453822
2: -1.9681013, -1.6588017, -1.9681013, -1.6588017, -0.0567770, 0.0567769
3: -0.7469844, -0.4570597, -0.7469844, -0.4570597, -0.0737288, 0.0737142
4: -2.7783232, -2.2692385, -2.7783232, -2.2692385, -0.1071590, 0.1071402
5: -0.2394420, 0.0989661, -0.2394420, 0.0989661, -0.0709230, 0.0709079
6: -0.1065651, 0.2169087, -0.1065651, 0.2169087, -0.2159095, 0.2159182
7: -0.7139419, -0.3450459, -0.7139419, -0.3450459, -0.0640284, 0.0640214
8: -2.5324945, -1.6453166, -2.5324945, -1.6453166, -0.6460805, 0.6461349
9: -1.5339894, -0.6906605, -1.5339894, -0.6906605, -0.4290919, 0.4291413

Time for backsubstitution: 5.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2829
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 259
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 3541

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 367

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261956, upper bound: 0.0260424
time: 13.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0260612, upper bound: 0.0261770
time: 203.40 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.2363062, -2.5159841, -3.2363062, -2.5159841, -0.2116241, 0.2115353
1: -1.2228723, -0.3951268, -1.2228723, -0.3951268, -0.2453822, 0.2452658
2: -1.9681013, -1.6588017, -1.9681013, -1.6588017, -0.0567769, 0.0567770
3: -0.7469844, -0.4570597, -0.7469844, -0.4570597, -0.0737142, 0.0737288
4: -2.7783232, -2.2692385, -2.7783232, -2.2692385, -0.1071402, 0.1071590
5: -0.2394420, 0.0989661, -0.2394420, 0.0989661, -0.0709079, 0.0709230
6: -0.1065651, 0.2169087, -0.1065651, 0.2169087, -0.2159183, 0.2159095
7: -0.7139419, -0.3450459, -0.7139419, -0.3450459, -0.0640214, 0.0640284
8: -2.5324945, -1.6453166, -2.5324945, -1.6453166, -0.6461353, 0.6460805
9: -1.5339894, -0.6906605, -1.5339894, -0.6906605, -0.4291413, 0.4290919

Time for backsubstitution: 5.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2829
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 259
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 3541

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 367

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261747, upper bound: 0.0260618
time: 83.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0260412, upper bound: 0.0261965
time: 8.53 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 97.92 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 97.92
Output dim: 3, lower bound: -0.0261956, upper bound: 0.0260424
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 97.92
Output dim: 3, lower bound: -0.0260612, upper bound: 0.0261770
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 97.92
Output dim: 3, lower bound: -0.0261747, upper bound: 0.0260618
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 97.92
Output dim: 3, lower bound: -0.0260412, upper bound: 0.0261965

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2363062, -2.5159841, -3.2363062, -2.5159841, -0.2102301, 0.2103160
1: -1.2228723, -0.3951268, -1.2228723, -0.3951268, -0.2449125, 0.2451032
2: -1.9681013, -1.6588017, -1.9681013, -1.6588017, -0.0528825, 0.0530571
3: -0.7469844, -0.4570597, -0.7469844, -0.4570597, -0.0721687, 0.0720200
4: -2.7783232, -2.2692385, -2.7783232, -2.2692385, -0.1021502, 0.1023161
5: -0.2394420, 0.0989661, -0.2394420, 0.0989661, -0.0687002, 0.0685121
6: -0.1065651, 0.2169087, -0.1065651, 0.2169087, -0.2168404, 0.2166506
7: -0.7139419, -0.3450459, -0.7139419, -0.3450459, -0.0584554, 0.0587426
8: -2.5324945, -1.6453166, -2.5324945, -1.6453166, -0.6450400, 0.6450489
9: -1.5339894, -0.6906605, -1.5339894, -0.6906605, -0.4255848, 0.4258468

Time for backsubstitution: 5.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2829
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 259
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 3541

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2618

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261695, upper bound: 0.0259819
time: 9.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261567, upper bound: 0.0259818
time: 131.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2363062, -2.5159841, -3.2363062, -2.5159841, -0.2102272, 0.2103190
1: -1.2228723, -0.3951268, -1.2228723, -0.3951268, -0.2449867, 0.2450289
2: -1.9681013, -1.6588017, -1.9681013, -1.6588017, -0.0530571, 0.0528825
3: -0.7469844, -0.4570597, -0.7469844, -0.4570597, -0.0720346, 0.0721540
4: -2.7783232, -2.2692385, -2.7783232, -2.2692385, -0.1023349, 0.1021314
5: -0.2394420, 0.0989661, -0.2394420, 0.0989661, -0.0685272, 0.0686851
6: -0.1065651, 0.2169087, -0.1065651, 0.2169087, -0.2166418, 0.2168492
7: -0.7139419, -0.3450459, -0.7139419, -0.3450459, -0.0587497, 0.0584484
8: -2.5324945, -1.6453166, -2.5324945, -1.6453166, -0.6449947, 0.6450946
9: -1.5339894, -0.6906605, -1.5339894, -0.6906605, -0.4257977, 0.4256341

Time for backsubstitution: 5.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2829
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 259
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 3541

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2618

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0260480, upper bound: 0.0261391
time: 117.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0260018, upper bound: 0.0261520
time: 5.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.2363062, -2.5159841, -3.2363062, -2.5159841, -0.2103189, 0.2102272
1: -1.2228723, -0.3951268, -1.2228723, -0.3951268, -0.2450289, 0.2449867
2: -1.9681013, -1.6588017, -1.9681013, -1.6588017, -0.0528825, 0.0530571
3: -0.7469844, -0.4570597, -0.7469844, -0.4570597, -0.0721540, 0.0720346
4: -2.7783232, -2.2692385, -2.7783232, -2.2692385, -0.1021314, 0.1023349
5: -0.2394420, 0.0989661, -0.2394420, 0.0989661, -0.0686851, 0.0685272
6: -0.1065651, 0.2169087, -0.1065651, 0.2169087, -0.2168492, 0.2166418
7: -0.7139419, -0.3450459, -0.7139419, -0.3450459, -0.0584484, 0.0587497
8: -2.5324945, -1.6453166, -2.5324945, -1.6453166, -0.6450949, 0.6449945
9: -1.5339894, -0.6906605, -1.5339894, -0.6906605, -0.4256339, 0.4257977

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2829
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 259
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 3541

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2618

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261500, upper bound: 0.0260026
time: 9.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0261368, upper bound: 0.0260494
time: 45.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.2363062, -2.5159841, -3.2363062, -2.5159841, -0.2103161, 0.2102301
1: -1.2228723, -0.3951268, -1.2228723, -0.3951268, -0.2451032, 0.2449126
2: -1.9681013, -1.6588017, -1.9681013, -1.6588017, -0.0530571, 0.0528825
3: -0.7469844, -0.4570597, -0.7469844, -0.4570597, -0.0720200, 0.0721687
4: -2.7783232, -2.2692385, -2.7783232, -2.2692385, -0.1023161, 0.1021502
5: -0.2394420, 0.0989661, -0.2394420, 0.0989661, -0.0685121, 0.0687002
6: -0.1065651, 0.2169087, -0.1065651, 0.2169087, -0.2166506, 0.2168404
7: -0.7139419, -0.3450459, -0.7139419, -0.3450459, -0.0587426, 0.0584554
8: -2.5324945, -1.6453166, -2.5324945, -1.6453166, -0.6450491, 0.6450403
9: -1.5339894, -0.6906605, -1.5339894, -0.6906605, -0.4258471, 0.4255848

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2829
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 259
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 3541

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2618

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0260285, upper bound: 0.0261594
time: 31.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0259804, upper bound: 0.0261711
time: 98.52 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 135.81 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 135.81
Output dim: 3, lower bound: -0.0261695, upper bound: 0.0259819
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 135.81
Output dim: 3, lower bound: -0.0261567, upper bound: 0.0259818
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 135.81
Output dim: 3, lower bound: -0.0260480, upper bound: 0.0261391
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 135.81
Output dim: 3, lower bound: -0.0260018, upper bound: 0.0261520
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 135.81
Output dim: 3, lower bound: -0.0261500, upper bound: 0.0260026
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 135.81
Output dim: 3, lower bound: -0.0261368, upper bound: 0.0260494
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 135.81
Output dim: 3, lower bound: -0.0260285, upper bound: 0.0261594
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 135.81
Output dim: 3, lower bound: -0.0259804, upper bound: 0.0261711

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2363062, -2.5159841, -3.2363062, -2.5159841, -0.2101168, 0.2102894
1: -1.2228723, -0.3951268, -1.2228723, -0.3951268, -0.2448135, 0.2450573
2: -1.9681013, -1.6588017, -1.9681013, -1.6588017, -0.0528802, 0.0530570
3: -0.7469844, -0.4570597, -0.7469844, -0.4570597, -0.0721435, 0.0719862
4: -2.7783232, -2.2692385, -2.7783232, -2.2692385, -0.1021111, 0.1022699
5: -0.2394420, 0.0989661, -0.2394420, 0.0989661, -0.0686765, 0.0684785
6: -0.1065651, 0.2169087, -0.1065651, 0.2169087, -0.2168301, 0.2166391
7: -0.7139419, -0.3450459, -0.7139419, -0.3450459, -0.0584500, 0.0587371
8: -2.5324945, -1.6453166, -2.5324945, -1.6453166, -0.6450062, 0.6450398
9: -1.5339894, -0.6906605, -1.5339894, -0.6906605, -0.4255533, 0.4258046

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2829
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 259
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 3541

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 547

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261515, upper bound: 0.0259704
time: 10.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261573, upper bound: 0.0259631
time: 122.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2363062, -2.5159841, -3.2363062, -2.5159841, -0.2102301, 0.2102028
1: -1.2228723, -0.3951268, -1.2228723, -0.3951268, -0.2449125, 0.2450041
2: -1.9681013, -1.6588017, -1.9681013, -1.6588017, -0.0528825, 0.0530548
3: -0.7469844, -0.4570597, -0.7469844, -0.4570597, -0.0721349, 0.0720200
4: -2.7783232, -2.2692385, -2.7783232, -2.2692385, -0.1021039, 0.1023161
5: -0.2394420, 0.0989661, -0.2394420, 0.0989661, -0.0686666, 0.0685121
6: -0.1065651, 0.2169087, -0.1065651, 0.2169087, -0.2168289, 0.2166506
7: -0.7139419, -0.3450459, -0.7139419, -0.3450459, -0.0584554, 0.0587372
8: -2.5324945, -1.6453166, -2.5324945, -1.6453166, -0.6450400, 0.6450150
9: -1.5339894, -0.6906605, -1.5339894, -0.6906605, -0.4255848, 0.4258156

Time for backsubstitution: 5.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2829
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 259
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 3541

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 547

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0261378, upper bound: 0.0260174
time: 147.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261452, upper bound: 0.0260104
time: 85.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.2363062, -2.5159841, -3.2363062, -2.5159841, -0.2102272, 0.2102057
1: -1.2228723, -0.3951268, -1.2228723, -0.3951268, -0.2449867, 0.2449299
2: -1.9681013, -1.6588017, -1.9681013, -1.6588017, -0.0530571, 0.0528802
3: -0.7469844, -0.4570597, -0.7469844, -0.4570597, -0.0720009, 0.0721540
4: -2.7783232, -2.2692385, -2.7783232, -2.2692385, -0.1022886, 0.1021314
5: -0.2394420, 0.0989661, -0.2394420, 0.0989661, -0.0684936, 0.0686851
6: -0.1065651, 0.2169087, -0.1065651, 0.2169087, -0.2166303, 0.2168492
7: -0.7139419, -0.3450459, -0.7139419, -0.3450459, -0.0587497, 0.0584430
8: -2.5324945, -1.6453166, -2.5324945, -1.6453166, -0.6449947, 0.6450608
9: -1.5339894, -0.6906605, -1.5339894, -0.6906605, -0.4257977, 0.4256024

Time for backsubstitution: 6.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2829
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 259
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 3541

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 547

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0259830, upper bound: 0.0261398
time: 9.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0259897, upper bound: 0.0261325
time: 39.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2363062, -2.5159841, -3.2363062, -2.5159841, -0.2102057, 0.2102020
1: -1.2228723, -0.3951268, -1.2228723, -0.3951268, -0.2449299, 0.2449418
2: -1.9681013, -1.6588017, -1.9681013, -1.6588017, -0.0528802, 0.0530570
3: -0.7469844, -0.4570597, -0.7469844, -0.4570597, -0.0721291, 0.0720009
4: -2.7783232, -2.2692385, -2.7783232, -2.2692385, -0.1020923, 0.1022886
5: -0.2394420, 0.0989661, -0.2394420, 0.0989661, -0.0686616, 0.0684936
6: -0.1065651, 0.2169087, -0.1065651, 0.2169087, -0.2168391, 0.2166303
7: -0.7139419, -0.3450459, -0.7139419, -0.3450459, -0.0584430, 0.0587442
8: -2.5324945, -1.6453166, -2.5324945, -1.6453166, -0.6450610, 0.6449850
9: -1.5339894, -0.6906605, -1.5339894, -0.6906605, -0.4256027, 0.4257557

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2829
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 259
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 3541

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 547

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0261316, upper bound: 0.0259909
time: 5.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0261381, upper bound: 0.0259838
time: 80.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.2363062, -2.5159841, -3.2363062, -2.5159841, -0.2102028, 0.2102236
1: -1.2228723, -0.3951268, -1.2228723, -0.3951268, -0.2450041, 0.2448770
2: -1.9681013, -1.6588017, -1.9681013, -1.6588017, -0.0530548, 0.0528837
3: -0.7469844, -0.4570597, -0.7469844, -0.4570597, -0.0720030, 0.0721349
4: -2.7783232, -2.2692385, -2.7783232, -2.2692385, -0.1022897, 0.1021039
5: -0.2394420, 0.0989661, -0.2394420, 0.0989661, -0.0684972, 0.0686667
6: -0.1065651, 0.2169087, -0.1065651, 0.2169087, -0.2166430, 0.2168289
7: -0.7139419, -0.3450459, -0.7139419, -0.3450459, -0.0587372, 0.0584521
8: -2.5324945, -1.6453166, -2.5324945, -1.6453166, -0.6450152, 0.6450427
9: -1.5339894, -0.6906605, -1.5339894, -0.6906605, -0.4258153, 0.4255509

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2829
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 259
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 3541

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 547

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0260102, upper bound: 0.0261473
time: 298.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0260169, upper bound: 0.0261395
time: 191.98 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 496.20 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 496.20
Output dim: 3, lower bound: -0.0261515, upper bound: 0.0259704
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 496.20
Output dim: 3, lower bound: -0.0261573, upper bound: 0.0259631
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 496.20
Output dim: 3, lower bound: -0.0261378, upper bound: 0.0260174
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 496.20
Output dim: 3, lower bound: -0.0261452, upper bound: 0.0260104
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 496.20
Output dim: 3, lower bound: -0.0259830, upper bound: 0.0261398
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 496.20
Output dim: 3, lower bound: -0.0259897, upper bound: 0.0261325
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 496.20
Output dim: 3, lower bound: -0.0261316, upper bound: 0.0259909
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 496.20
Output dim: 3, lower bound: -0.0261381, upper bound: 0.0259838
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 496.20
Output dim: 3, lower bound: -0.0260102, upper bound: 0.0261473
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 496.20
Output dim: 3, lower bound: -0.0260169, upper bound: 0.0261395
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 496.20
Output dim: 3, lower bound: -0.0259804, upper bound: 0.0261711

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 202.47 + 1920.00 = 2122.46 seconds
