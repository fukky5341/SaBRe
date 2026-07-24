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
execution time: IAR + RelationalAnalysis = 8.40 + 179.56 = 187.96 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0262224, upper bound: 0.0262229

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 910
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 259
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2829
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 2071

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 910

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262213, upper bound: 0.0262218
time: 114.20 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262215, upper bound: 0.0262223
time: 182.73 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 296.94 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 296.94
Output dim: 3, lower bound: -0.0262213, upper bound: 0.0262218
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 296.94
Output dim: 3, lower bound: -0.0262215, upper bound: 0.0262223

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -3.2363062, -2.5159841, -3.2363062, -2.5159841, -0.2151082, 0.2151082
1: -1.2228723, -0.3951268, -1.2228723, -0.3951268, -0.2485532, 0.2485532
2: -1.9681013, -1.6588017, -1.9681013, -1.6588017, -0.0567758, 0.0567758
3: -0.7469844, -0.4570597, -0.7469844, -0.4570597, -0.0747739, 0.0747739
4: -2.7783232, -2.2692385, -2.7783232, -2.2692385, -0.1083375, 0.1083375
5: -0.2394420, 0.0989661, -0.2394420, 0.0989661, -0.0720247, 0.0720247
6: -0.1065651, 0.2169087, -0.1065651, 0.2169087, -0.2161174, 0.2161173
7: -0.7139419, -0.3450459, -0.7139419, -0.3450459, -0.0640447, 0.0640447
8: -2.5324945, -1.6453166, -2.5324945, -1.6453166, -0.6475115, 0.6475120
9: -1.5339894, -0.6906605, -1.5339894, -0.6906605, -0.4303198, 0.4303198

Time for backsubstitution: 5.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 259
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 2829
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2350

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 908

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262212, upper bound: 0.0262226
time: 78.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262207, upper bound: 0.0262227
time: 41.42 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -3.2363062, -2.5159841, -3.2363062, -2.5159841, -0.2151082, 0.2151082
1: -1.2228723, -0.3951268, -1.2228723, -0.3951268, -0.2485532, 0.2485532
2: -1.9681013, -1.6588017, -1.9681013, -1.6588017, -0.0567758, 0.0567758
3: -0.7469844, -0.4570597, -0.7469844, -0.4570597, -0.0747739, 0.0747739
4: -2.7783232, -2.2692385, -2.7783232, -2.2692385, -0.1083375, 0.1083375
5: -0.2394420, 0.0989661, -0.2394420, 0.0989661, -0.0720247, 0.0720247
6: -0.1065651, 0.2169087, -0.1065651, 0.2169087, -0.2161174, 0.2161173
7: -0.7139419, -0.3450459, -0.7139419, -0.3450459, -0.0640447, 0.0640447
8: -2.5324945, -1.6453166, -2.5324945, -1.6453166, -0.6475115, 0.6475120
9: -1.5339894, -0.6906605, -1.5339894, -0.6906605, -0.4303198, 0.4303198

Time for backsubstitution: 5.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 259
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2829
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 973

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 319

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261803, upper bound: 0.0262182
time: 139.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262210, upper bound: 0.0261833
time: 99.68 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 244.72 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 244.72
Output dim: 3, lower bound: -0.0262212, upper bound: 0.0262226
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 244.72
Output dim: 3, lower bound: -0.0262207, upper bound: 0.0262227
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 244.72
Output dim: 3, lower bound: -0.0261803, upper bound: 0.0262182
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 244.72
Output dim: 3, lower bound: -0.0262210, upper bound: 0.0261833

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2363062, -2.5159841, -3.2363062, -2.5159841, -0.2151081, 0.2151082
1: -1.2228723, -0.3951268, -1.2228723, -0.3951268, -0.2485531, 0.2485532
2: -1.9681013, -1.6588017, -1.9681013, -1.6588017, -0.0567758, 0.0567758
3: -0.7469844, -0.4570597, -0.7469844, -0.4570597, -0.0747738, 0.0747738
4: -2.7783232, -2.2692385, -2.7783232, -2.2692385, -0.1083375, 0.1083375
5: -0.2394420, 0.0989661, -0.2394420, 0.0989661, -0.0720246, 0.0720246
6: -0.1065651, 0.2169087, -0.1065651, 0.2169087, -0.2161175, 0.2161174
7: -0.7139419, -0.3450459, -0.7139419, -0.3450459, -0.0640447, 0.0640447
8: -2.5324945, -1.6453166, -2.5324945, -1.6453166, -0.6475115, 0.6475120
9: -1.5339894, -0.6906605, -1.5339894, -0.6906605, -0.4303198, 0.4303198

Time for backsubstitution: 5.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2829
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 259
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2634

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 99

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261237, upper bound: 0.0262228
time: 5.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262196, upper bound: 0.0261280
time: 78.70 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2363062, -2.5159841, -3.2363062, -2.5159841, -0.2151081, 0.2151082
1: -1.2228723, -0.3951268, -1.2228723, -0.3951268, -0.2485531, 0.2485532
2: -1.9681013, -1.6588017, -1.9681013, -1.6588017, -0.0567758, 0.0567758
3: -0.7469844, -0.4570597, -0.7469844, -0.4570597, -0.0747738, 0.0747738
4: -2.7783232, -2.2692385, -2.7783232, -2.2692385, -0.1083375, 0.1083375
5: -0.2394420, 0.0989661, -0.2394420, 0.0989661, -0.0720246, 0.0720246
6: -0.1065651, 0.2169087, -0.1065651, 0.2169087, -0.2161175, 0.2161174
7: -0.7139419, -0.3450459, -0.7139419, -0.3450459, -0.0640447, 0.0640447
8: -2.5324945, -1.6453166, -2.5324945, -1.6453166, -0.6475115, 0.6475120
9: -1.5339894, -0.6906605, -1.5339894, -0.6906605, -0.4303198, 0.4303198

Time for backsubstitution: 5.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2829
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 259
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 502

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 547

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262037, upper bound: 0.0262114
time: 8.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262095, upper bound: 0.0262038
time: 129.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.2363062, -2.5159841, -3.2363062, -2.5159841, -0.2155767, 0.2155640
1: -1.2228723, -0.3951268, -1.2228723, -0.3951268, -0.2454428, 0.2452787
2: -1.9681013, -1.6588017, -1.9681013, -1.6588017, -0.0548893, 0.0549855
3: -0.7469844, -0.4570597, -0.7469844, -0.4570597, -0.0746034, 0.0746092
4: -2.7783232, -2.2692385, -2.7783232, -2.2692385, -0.1060560, 0.1061758
5: -0.2394420, 0.0989661, -0.2394420, 0.0989661, -0.0722260, 0.0722167
6: -0.1065651, 0.2169087, -0.1065651, 0.2169087, -0.2103999, 0.2106894
7: -0.7139419, -0.3450459, -0.7139419, -0.3450459, -0.0603792, 0.0602031
8: -2.5324945, -1.6453166, -2.5324945, -1.6453166, -0.6478052, 0.6478338
9: -1.5339894, -0.6906605, -1.5339894, -0.6906605, -0.4286237, 0.4285333

Time for backsubstitution: 5.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2829
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 259
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 547

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2071

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261826, upper bound: 0.0262227
time: 9.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261825, upper bound: 0.0262215
time: 8.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.2363062, -2.5159841, -3.2363062, -2.5159841, -0.2155640, 0.2155768
1: -1.2228723, -0.3951268, -1.2228723, -0.3951268, -0.2452788, 0.2454427
2: -1.9681013, -1.6588017, -1.9681013, -1.6588017, -0.0549855, 0.0548893
3: -0.7469844, -0.4570597, -0.7469844, -0.4570597, -0.0746092, 0.0746034
4: -2.7783232, -2.2692385, -2.7783232, -2.2692385, -0.1061758, 0.1060560
5: -0.2394420, 0.0989661, -0.2394420, 0.0989661, -0.0722167, 0.0722260
6: -0.1065651, 0.2169087, -0.1065651, 0.2169087, -0.2106894, 0.2104000
7: -0.7139419, -0.3450459, -0.7139419, -0.3450459, -0.0602031, 0.0603792
8: -2.5324945, -1.6453166, -2.5324945, -1.6453166, -0.6478343, 0.6478047
9: -1.5339894, -0.6906605, -1.5339894, -0.6906605, -0.4285333, 0.4286237

Time for backsubstitution: 5.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 259
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2829
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2566

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 259

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262195, upper bound: 0.0261543
time: 9.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261913, upper bound: 0.0261831
time: 11.08 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 25.84 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 25.84
Output dim: 3, lower bound: -0.0261237, upper bound: 0.0262228
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 25.84
Output dim: 3, lower bound: -0.0262196, upper bound: 0.0261280
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 25.84
Output dim: 3, lower bound: -0.0262037, upper bound: 0.0262114
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 25.84
Output dim: 3, lower bound: -0.0262095, upper bound: 0.0262038
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 25.84
Output dim: 3, lower bound: -0.0261826, upper bound: 0.0262227
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 25.84
Output dim: 3, lower bound: -0.0261825, upper bound: 0.0262215
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 25.84
Output dim: 3, lower bound: -0.0262195, upper bound: 0.0261543
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 25.84
Output dim: 3, lower bound: -0.0261913, upper bound: 0.0261831

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2363062, -2.5159841, -3.2363062, -2.5159841, -0.2136652, 0.2134039
1: -1.2228723, -0.3951268, -1.2228723, -0.3951268, -0.2471506, 0.2469077
2: -1.9681013, -1.6588017, -1.9681013, -1.6588017, -0.0567694, 0.0567697
3: -0.7469844, -0.4570597, -0.7469844, -0.4570597, -0.0742584, 0.0743396
4: -2.7783232, -2.2692385, -2.7783232, -2.2692385, -0.1082908, 0.1082945
5: -0.2394420, 0.0989661, -0.2394420, 0.0989661, -0.0714520, 0.0715160
6: -0.1065651, 0.2169087, -0.1065651, 0.2169087, -0.2155682, 0.2156399
7: -0.7139419, -0.3450459, -0.7139419, -0.3450459, -0.0639776, 0.0639728
8: -2.5324945, -1.6453166, -2.5324945, -1.6453166, -0.6469736, 0.6469340
9: -1.5339894, -0.6906605, -1.5339894, -0.6906605, -0.4298656, 0.4298117

Time for backsubstitution: 5.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 259
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2829
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 909

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0260921, upper bound: 0.0261862
time: 116.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261129, upper bound: 0.0261871
time: 32.12 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -3.2363062, -2.5159841, -3.2363062, -2.5159841, -0.2134039, 0.2136653
1: -1.2228723, -0.3951268, -1.2228723, -0.3951268, -0.2469077, 0.2471507
2: -1.9681013, -1.6588017, -1.9681013, -1.6588017, -0.0567697, 0.0567694
3: -0.7469844, -0.4570597, -0.7469844, -0.4570597, -0.0743396, 0.0742584
4: -2.7783232, -2.2692385, -2.7783232, -2.2692385, -0.1082945, 0.1082908
5: -0.2394420, 0.0989661, -0.2394420, 0.0989661, -0.0715160, 0.0714520
6: -0.1065651, 0.2169087, -0.1065651, 0.2169087, -0.2156399, 0.2155682
7: -0.7139419, -0.3450459, -0.7139419, -0.3450459, -0.0639728, 0.0639776
8: -2.5324945, -1.6453166, -2.5324945, -1.6453166, -0.6469340, 0.6469741
9: -1.5339894, -0.6906605, -1.5339894, -0.6906605, -0.4298117, 0.4298656

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 2829
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 259
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 679

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2850

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262186, upper bound: 0.0261223
time: 105.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262149, upper bound: 0.0261258
time: 49.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -3.2363062, -2.5159841, -3.2363062, -2.5159841, -0.2150562, 0.2150552
1: -1.2228723, -0.3951268, -1.2228723, -0.3951268, -0.2479794, 0.2479694
2: -1.9681013, -1.6588017, -1.9681013, -1.6588017, -0.0540530, 0.0538923
3: -0.7469844, -0.4570597, -0.7469844, -0.4570597, -0.0741951, 0.0741940
4: -2.7783232, -2.2692385, -2.7783232, -2.2692385, -0.1074625, 0.1073800
5: -0.2394420, 0.0989661, -0.2394420, 0.0989661, -0.0713171, 0.0713214
6: -0.1065651, 0.2169087, -0.1065651, 0.2169087, -0.2150710, 0.2151358
7: -0.7139419, -0.3450459, -0.7139419, -0.3450459, -0.0618911, 0.0618005
8: -2.5324945, -1.6453166, -2.5324945, -1.6453166, -0.6464744, 0.6464734
9: -1.5339894, -0.6906605, -1.5339894, -0.6906605, -0.4297578, 0.4297559

Time for backsubstitution: 5.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 259
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 2829
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2055

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 502

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262014, upper bound: 0.0262113
time: 4.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261998, upper bound: 0.0262108
time: 45.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -3.2363062, -2.5159841, -3.2363062, -2.5159841, -0.2150552, 0.2150562
1: -1.2228723, -0.3951268, -1.2228723, -0.3951268, -0.2479694, 0.2479794
2: -1.9681013, -1.6588017, -1.9681013, -1.6588017, -0.0538923, 0.0540530
3: -0.7469844, -0.4570597, -0.7469844, -0.4570597, -0.0741940, 0.0741952
4: -2.7783232, -2.2692385, -2.7783232, -2.2692385, -0.1073800, 0.1074625
5: -0.2394420, 0.0989661, -0.2394420, 0.0989661, -0.0713214, 0.0713171
6: -0.1065651, 0.2169087, -0.1065651, 0.2169087, -0.2151358, 0.2150709
7: -0.7139419, -0.3450459, -0.7139419, -0.3450459, -0.0618005, 0.0618911
8: -2.5324945, -1.6453166, -2.5324945, -1.6453166, -0.6464734, 0.6464744
9: -1.5339894, -0.6906605, -1.5339894, -0.6906605, -0.4297559, 0.4297578

Time for backsubstitution: 5.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 259
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 912
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 319
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2829
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 942

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 958

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262092, upper bound: 0.0262044
time: 53.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262103, upper bound: 0.0262040
time: 113.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -3.2363062, -2.5159841, -3.2363062, -2.5159841, -0.2152016, 0.2151704
1: -1.2228723, -0.3951268, -1.2228723, -0.3951268, -0.2454331, 0.2452686
2: -1.9681013, -1.6588017, -1.9681013, -1.6588017, -0.0548586, 0.0549546
3: -0.7469844, -0.4570597, -0.7469844, -0.4570597, -0.0746125, 0.0746175
4: -2.7783232, -2.2692385, -2.7783232, -2.2692385, -0.1059204, 0.1060334
5: -0.2394420, 0.0989661, -0.2394420, 0.0989661, -0.0722267, 0.0722173
6: -0.1065651, 0.2169087, -0.1065651, 0.2169087, -0.2102460, 0.2105440
7: -0.7139419, -0.3450459, -0.7139419, -0.3450459, -0.0603773, 0.0602012
8: -2.5324945, -1.6453166, -2.5324945, -1.6453166, -0.6474075, 0.6474128
9: -1.5339894, -0.6906605, -1.5339894, -0.6906605, -0.4282153, 0.4281437

Time for backsubstitution: 5.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2850
type: DSZ, layer: 1, pos: 2055
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 501
type: DSZ, layer: 1, pos: 2829
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 2889
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 3496
type: DSZ, layer: 1, pos: 679
type: DSZ, layer: 1, pos: 3079
type: DSZ, layer: 1, pos: 943
type: DSZ, layer: 1, pos: 973
type: DSZ, layer: 1, pos: 2351
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2828
type: DSZ, layer: 1, pos: 925
type: DSZ, layer: 1, pos: 3318
type: DSZ, layer: 1, pos: 2634
type: DSZ, layer: 1, pos: 942
type: DSZ, layer: 1, pos: 2832
type: DSZ, layer: 1, pos: 929
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 958
type: DSZ, layer: 1, pos: 2507
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2649
type: DSZ, layer: 1, pos: 2314
type: DSZ, layer: 1, pos: 2097
type: DSZ, layer: 1, pos: 930
type: DSZ, layer: 1, pos: 927
type: DSZ, layer: 1, pos: 974
type: DSZ, layer: 1, pos: 2801
type: DSZ, layer: 1, pos: 2831
type: DSZ, layer: 1, pos: 3236
type: DSZ, layer: 1, pos: 902
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 529
type: DSZ, layer: 1, pos: 959
type: DSZ, layer: 1, pos: 913
type: DSZ, layer: 1, pos: 900
type: DSZ, layer: 1, pos: 911
type: DSZ, layer: 1, pos: 745
type: DSZ, layer: 1, pos: 2431
type: DSZ, layer: 1, pos: 926
type: DSZ, layer: 1, pos: 931
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2669
type: DSZ, layer: 1, pos: 2056
type: DSZ, layer: 1, pos: 3441
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 261
type: DSZ, layer: 1, pos: 3216
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 928
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 917
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2350
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 901
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2819
type: DSZ, layer: 1, pos: 2150
type: DSZ, layer: 1, pos: 3541
type: DSZ, layer: 1, pos: 3130
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 916
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2627
type: DSZ, layer: 1, pos: 944
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 259
type: DSZ, layer: 1, pos: 738
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 502
type: DSZ, layer: 1, pos: 374
type: DSZ, layer: 1, pos: 3501
type: DSZ, layer: 1, pos: 378
type: DSZ, layer: 1, pos: 909
type: DSZ, layer: 1, pos: 2405
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 907
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 914
type: DSZ, layer: 1, pos: 557
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 2152
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 915
type: DSZ, layer: 1, pos: 908
type: DSZ, layer: 1, pos: 2403
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2800
type: DSZ, layer: 1, pos: 2573
type: DSZ, layer: 1, pos: 2891
type: DSZ, layer: 1, pos: 3129
type: DSZ, layer: 1, pos: 65
type: DSZ, layer: 1, pos: 972
type: DSZ, layer: 1, pos: 192
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2544
type: DSZ, layer: 1, pos: 912

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 385

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261807, upper bound: 0.0258303
time: 94.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0258307, upper bound: 0.0262188
time: 29.43 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 129.44 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 129.44
Output dim: 3, lower bound: -0.0260921, upper bound: 0.0261862
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 129.44
Output dim: 3, lower bound: -0.0261129, upper bound: 0.0261871
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 129.44
Output dim: 3, lower bound: -0.0262186, upper bound: 0.0261223
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 129.44
Output dim: 3, lower bound: -0.0262149, upper bound: 0.0261258
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 129.44
Output dim: 3, lower bound: -0.0262014, upper bound: 0.0262113
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 129.44
Output dim: 3, lower bound: -0.0261998, upper bound: 0.0262108
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 129.44
Output dim: 3, lower bound: -0.0262092, upper bound: 0.0262044
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 129.44
Output dim: 3, lower bound: -0.0262103, upper bound: 0.0262040
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 129.44
Output dim: 3, lower bound: -0.0261807, upper bound: 0.0258303
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 129.44
Output dim: 3, lower bound: -0.0258307, upper bound: 0.0262188
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 129.44
Output dim: 3, lower bound: -0.0261825, upper bound: 0.0262215
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 129.44
Output dim: 3, lower bound: -0.0262195, upper bound: 0.0261543
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 129.44
Output dim: 3, lower bound: -0.0261913, upper bound: 0.0261831

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 187.96 + 1621.80 = 1809.77 seconds
