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
execution time: IAR + RelationalAnalysis = 7.14 + 180.97 = 188.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0262224, upper bound: 0.0262229

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 385
type: B, layer: 1, pos: 385
type: A, layer: 1, pos: 378
type: B, layer: 1, pos: 378
type: A, layer: 1, pos: 367
type: B, layer: 1, pos: 367
type: A, layer: 1, pos: 2350
type: B, layer: 1, pos: 2350
type: A, layer: 1, pos: 2152
type: B, layer: 1, pos: 2152
type: A, layer: 1, pos: 2351
type: B, layer: 1, pos: 2351
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2405
type: B, layer: 1, pos: 2405
type: A, layer: 1, pos: 3079
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 2544
type: B, layer: 1, pos: 2544
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 2604
type: B, layer: 1, pos: 2604
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 261
type: B, layer: 1, pos: 261
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2649
type: B, layer: 1, pos: 2649
type: A, layer: 1, pos: 259
type: B, layer: 1, pos: 259
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 3441
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 3130
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 2097
type: B, layer: 1, pos: 2097
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: A, layer: 1, pos: 3496
type: B, layer: 1, pos: 3496
type: B, layer: 1, pos: 2431
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2801
type: B, layer: 1, pos: 2801
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 3129
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 2627
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2829
type: B, layer: 1, pos: 2829
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 2573
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 2831
type: B, layer: 1, pos: 2831
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 3318
type: B, layer: 1, pos: 3318
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 3541
type: B, layer: 1, pos: 3541
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 2850
type: B, layer: 1, pos: 2850
type: A, layer: 1, pos: 2088
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 319
type: A, layer: 1, pos: 319
type: A, layer: 1, pos: 3236
type: B, layer: 1, pos: 3236
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 2150
type: B, layer: 1, pos: 2150
type: A, layer: 1, pos: 2776
type: B, layer: 1, pos: 2776
type: A, layer: 1, pos: 2832
type: B, layer: 1, pos: 2832
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: A, layer: 1, pos: 2656
type: B, layer: 1, pos: 2656
type: A, layer: 1, pos: 192
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 3501
type: A, layer: 1, pos: 3501
type: A, layer: 1, pos: 3266
type: B, layer: 1, pos: 3266
type: A, layer: 1, pos: 2891
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2805
type: A, layer: 1, pos: 2805
type: A, layer: 1, pos: 2889
type: B, layer: 1, pos: 2889
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2507
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 2033
type: B, layer: 1, pos: 2033
type: A, layer: 1, pos: 2442
type: B, layer: 1, pos: 2442
type: A, layer: 1, pos: 2067
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 3426
type: A, layer: 1, pos: 3426
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 2055
type: B, layer: 1, pos: 2055
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 501
type: A, layer: 1, pos: 501
type: B, layer: 1, pos: 2056
type: A, layer: 1, pos: 2056
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 3123
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 374
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 2669
type: A, layer: 1, pos: 2819
type: B, layer: 1, pos: 374
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 2669
type: B, layer: 1, pos: 2819

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 385

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262193, upper bound: 0.0258713
time: 22.88 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262196, upper bound: 0.0262210
time: 73.08 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 96.04 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 96.04
Output dim: 3, lower bound: -0.0262193, upper bound: 0.0258713
NS_A2, status: Status.UNKNOWN, split count: 1, time: 96.04
Output dim: 3, lower bound: -0.0262196, upper bound: 0.0262210

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3.2355075, -2.5159979, -3.2356565, -2.5159955, -0.2142111, 0.2143608
1: -1.2215176, -0.3951297, -1.2217712, -0.3951287, -0.2471109, 0.2473543
2: -1.9680954, -1.6597381, -1.9680963, -1.6595629, -0.0558115, 0.0556158
3: -0.7469041, -0.4594111, -0.7469160, -0.4589741, -0.0725874, 0.0721228
4: -2.7781565, -2.2692432, -2.7781870, -2.2692425, -0.1082102, 0.1082309
5: -0.2394345, 0.0966150, -0.2394356, 0.0970505, -0.0699938, 0.0695456
6: -0.1058037, 0.2169067, -0.1059288, 0.2169071, -0.2153239, 0.2154495
7: -0.7135994, -0.3450603, -0.7136354, -0.3450579, -0.0633302, 0.0633977
8: -2.5324917, -1.6455708, -2.5324926, -1.6455231, -0.6473031, 0.6472545
9: -1.5325480, -0.6906619, -1.5328140, -0.6906619, -0.4287107, 0.4289887

Time for backsubstitution: 5.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 378
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 367
type: A, layer: 1, pos: 367
type: B, layer: 1, pos: 2350
type: A, layer: 1, pos: 2350
type: B, layer: 1, pos: 2152
type: A, layer: 1, pos: 2152
type: A, layer: 1, pos: 2351
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2405
type: B, layer: 1, pos: 2405
type: A, layer: 1, pos: 3079
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 2544
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 2604
type: B, layer: 1, pos: 2604
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 261
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2649
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 259
type: A, layer: 1, pos: 259
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 3441
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 3130
type: A, layer: 1, pos: 3130
type: B, layer: 1, pos: 2097
type: A, layer: 1, pos: 2097
type: B, layer: 1, pos: 2637
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 3496
type: A, layer: 1, pos: 3496
type: B, layer: 1, pos: 2431
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2801
type: B, layer: 1, pos: 2801
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 3129
type: A, layer: 1, pos: 3129
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 3033
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 2627
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2829
type: B, layer: 1, pos: 2829
type: B, layer: 1, pos: 2573
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 2831
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 3318
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 3541
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 2850
type: B, layer: 1, pos: 2850
type: A, layer: 1, pos: 2088
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 319
type: A, layer: 1, pos: 319
type: B, layer: 1, pos: 3236
type: A, layer: 1, pos: 3236
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 2150
type: B, layer: 1, pos: 2776
type: A, layer: 1, pos: 2776
type: B, layer: 1, pos: 2150
type: A, layer: 1, pos: 2832
type: B, layer: 1, pos: 2832
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: A, layer: 1, pos: 2656
type: B, layer: 1, pos: 2656
type: A, layer: 1, pos: 192
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 3501
type: A, layer: 1, pos: 3501
type: A, layer: 1, pos: 3266
type: B, layer: 1, pos: 3266
type: A, layer: 1, pos: 2891
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2805
type: A, layer: 1, pos: 2805
type: A, layer: 1, pos: 2889
type: B, layer: 1, pos: 2889
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2507
type: A, layer: 1, pos: 2507
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2033
type: A, layer: 1, pos: 2033
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2067
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 2442
type: B, layer: 1, pos: 3426
type: A, layer: 1, pos: 3426
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 2055
type: B, layer: 1, pos: 2055
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 501
type: A, layer: 1, pos: 501
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 2056
type: A, layer: 1, pos: 2056
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 928
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 374
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 2669
type: A, layer: 1, pos: 2819
type: B, layer: 1, pos: 374
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 2669
type: B, layer: 1, pos: 2819
type: B, layer: 1, pos: 907

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 378

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262055, upper bound: 0.0257298
time: 53.49 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262051, upper bound: 0.0258575
time: 16.47 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -3.2361240, -2.5150695, -3.2361600, -2.5159836, -0.2142537, 0.2161170
1: -1.2226720, -0.3935633, -1.2227120, -0.3951268, -0.2471300, 0.2500807
2: -1.9692461, -1.6589578, -1.9680996, -1.6589310, -0.0582842, 0.0557028
3: -0.7502308, -0.4571087, -0.7469834, -0.4571037, -0.0784058, 0.0724673
4: -2.7781153, -2.2691021, -2.7781413, -2.2692385, -0.1082183, 0.1082282
5: -0.2424492, 0.0989411, -0.2394419, 0.0989460, -0.0751227, 0.0698041
6: -0.1066065, 0.2174044, -0.1065582, 0.2169087, -0.2159501, 0.2165870
7: -0.7135988, -0.3448235, -0.7136527, -0.3450460, -0.0633999, 0.0649096
8: -2.5323052, -1.6461320, -2.5324917, -1.6460166, -0.6466589, 0.6468995
9: -1.5339346, -0.6891170, -1.5338221, -0.6906619, -0.4287992, 0.4312696

Time for backsubstitution: 5.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 378
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 367
type: A, layer: 1, pos: 367
type: B, layer: 1, pos: 2350
type: A, layer: 1, pos: 2350
type: B, layer: 1, pos: 2152
type: A, layer: 1, pos: 2152
type: A, layer: 1, pos: 2351
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2405
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: B, layer: 1, pos: 2544
type: A, layer: 1, pos: 2544
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 2604
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 261
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2649
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 259
type: A, layer: 1, pos: 259
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 3441
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 3130
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 3130
type: B, layer: 1, pos: 2097
type: A, layer: 1, pos: 2097
type: B, layer: 1, pos: 2637
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 3496
type: A, layer: 1, pos: 3496
type: B, layer: 1, pos: 2431
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2801
type: B, layer: 1, pos: 2801
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 3129
type: A, layer: 1, pos: 3129
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 3033
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 2627
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2829
type: B, layer: 1, pos: 2829
type: B, layer: 1, pos: 2573
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 2831
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 3318
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 3541
type: B, layer: 1, pos: 3541
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 2850
type: B, layer: 1, pos: 2850
type: A, layer: 1, pos: 2088
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 319
type: B, layer: 1, pos: 3236
type: A, layer: 1, pos: 3236
type: A, layer: 1, pos: 319
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 2776
type: A, layer: 1, pos: 2776
type: B, layer: 1, pos: 2150
type: A, layer: 1, pos: 2832
type: B, layer: 1, pos: 2832
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: A, layer: 1, pos: 192
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 3501
type: A, layer: 1, pos: 3501
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 3266
type: B, layer: 1, pos: 3266
type: B, layer: 1, pos: 2805
type: A, layer: 1, pos: 2805
type: B, layer: 1, pos: 2891
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2507
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2033
type: A, layer: 1, pos: 2033
type: B, layer: 1, pos: 2067
type: A, layer: 1, pos: 2067
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 2055
type: B, layer: 1, pos: 2055
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 3426
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2056
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2056
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 374
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 2669
type: A, layer: 1, pos: 2819
type: B, layer: 1, pos: 374
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 2669
type: B, layer: 1, pos: 2819
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 2071
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 3123

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 378

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262050, upper bound: 0.0260787
time: 104.55 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262051, upper bound: 0.0262071
time: 130.71 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 240.88 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 240.88
Output dim: 3, lower bound: -0.0262055, upper bound: 0.0257298
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 240.88
Output dim: 3, lower bound: -0.0262051, upper bound: 0.0258575
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 240.88
Output dim: 3, lower bound: -0.0262050, upper bound: 0.0260787
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 240.88
Output dim: 3, lower bound: -0.0262051, upper bound: 0.0262071

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -3.2354963, -2.5163536, -3.2356486, -2.5162606, -0.2138760, 0.2139193
1: -1.2214589, -0.3951302, -1.2217264, -0.3951287, -0.2470189, 0.2472724
2: -1.9680936, -1.6601710, -1.9680948, -1.6599038, -0.0553263, 0.0550050
3: -0.7468504, -0.4598987, -0.7468755, -0.4593521, -0.0718652, 0.0712703
4: -2.7781515, -2.2693236, -2.7781835, -2.2693019, -0.1080960, 0.1080896
5: -0.2394290, 0.0960983, -0.2394314, 0.0966527, -0.0693110, 0.0687049
6: -0.1053599, 0.2169061, -0.1055974, 0.2169064, -0.2148774, 0.2151156
7: -0.7135952, -0.3451341, -0.7136322, -0.3451129, -0.0632186, 0.0632512
8: -2.5324895, -1.6462688, -2.5324903, -1.6460395, -0.6468310, 0.6466296
9: -1.5323329, -0.6906624, -1.5326548, -0.6906624, -0.4283814, 0.4287052

Time for backsubstitution: 5.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 367
type: A, layer: 1, pos: 367
type: B, layer: 1, pos: 2350
type: A, layer: 1, pos: 2350
type: B, layer: 1, pos: 2152
type: A, layer: 1, pos: 2152
type: A, layer: 1, pos: 2351
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2405
type: B, layer: 1, pos: 2405
type: A, layer: 1, pos: 3079
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 2544
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 2604
type: B, layer: 1, pos: 2604
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 261
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2649
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 259
type: A, layer: 1, pos: 259
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 3441
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 3130
type: A, layer: 1, pos: 3130
type: B, layer: 1, pos: 2097
type: A, layer: 1, pos: 2097
type: B, layer: 1, pos: 2637
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3496
type: B, layer: 1, pos: 3496
type: B, layer: 1, pos: 2431
type: A, layer: 1, pos: 2431
type: B, layer: 1, pos: 378
type: A, layer: 1, pos: 2801
type: B, layer: 1, pos: 2801
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 3129
type: A, layer: 1, pos: 3129
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 3033
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 2627
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2829
type: B, layer: 1, pos: 2829
type: B, layer: 1, pos: 2573
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 2831
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 3318
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 3541
type: A, layer: 1, pos: 3541
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2850
type: B, layer: 1, pos: 2850
type: A, layer: 1, pos: 2088
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 319
type: A, layer: 1, pos: 319
type: A, layer: 1, pos: 3236
type: B, layer: 1, pos: 3236
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 2150
type: B, layer: 1, pos: 2776
type: A, layer: 1, pos: 2776
type: B, layer: 1, pos: 2150
type: A, layer: 1, pos: 2832
type: B, layer: 1, pos: 2832
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: A, layer: 1, pos: 2656
type: B, layer: 1, pos: 2656
type: A, layer: 1, pos: 192
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 3501
type: A, layer: 1, pos: 3501
type: A, layer: 1, pos: 3266
type: B, layer: 1, pos: 3266
type: A, layer: 1, pos: 2891
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2805
type: A, layer: 1, pos: 2805
type: A, layer: 1, pos: 2889
type: B, layer: 1, pos: 2889
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2507
type: A, layer: 1, pos: 2507
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 2033
type: A, layer: 1, pos: 2033
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2067
type: A, layer: 1, pos: 2067
type: B, layer: 1, pos: 3426
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 3426
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 2055
type: B, layer: 1, pos: 2055
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 501
type: A, layer: 1, pos: 501
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 972
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 911
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 913
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 374
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 2669
type: A, layer: 1, pos: 2819
type: B, layer: 1, pos: 374
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 2669
type: B, layer: 1, pos: 2819
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 2071
type: A, layer: 1, pos: 3123

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 367

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0260673, upper bound: 0.0257240
time: 162.29 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262031, upper bound: 0.0257267
time: 440.24 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -3.2374601, -2.5162506, -3.2356553, -2.5162473, -0.2160260, 0.2140556
1: -1.2212672, -0.3950214, -1.2215590, -0.3951292, -0.2470792, 0.2472837
2: -1.9700000, -1.6599603, -1.9680955, -1.6597350, -0.0580126, 0.0551082
3: -0.7495751, -0.4596704, -0.7469137, -0.4592600, -0.0759463, 0.0715909
4: -2.7785716, -2.2694578, -2.7781868, -2.2695022, -0.1088282, 0.1082437
5: -0.2419692, 0.0962668, -0.2394355, 0.0967259, -0.0731282, 0.0688669
6: -0.1058526, 0.2173167, -0.1059175, 0.2169073, -0.2152687, 0.2157632
7: -0.7148478, -0.3455109, -0.7136354, -0.3454118, -0.0650184, 0.0632583
8: -2.5351381, -1.6458998, -2.5324907, -1.6458454, -0.6494236, 0.6467016
9: -1.5319443, -0.6904650, -1.5321159, -0.6906624, -0.4289351, 0.4283311

Time for backsubstitution: 5.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 367
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 2350
type: A, layer: 1, pos: 2350
type: B, layer: 1, pos: 2152
type: A, layer: 1, pos: 2152
type: A, layer: 1, pos: 2351
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2405
type: B, layer: 1, pos: 2405
type: A, layer: 1, pos: 3079
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 2544
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 2604
type: B, layer: 1, pos: 2604
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 261
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2649
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 259
type: A, layer: 1, pos: 259
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 3441
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 378
type: A, layer: 1, pos: 3130
type: B, layer: 1, pos: 2097
type: A, layer: 1, pos: 2097
type: B, layer: 1, pos: 2637
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 3496
type: A, layer: 1, pos: 3496
type: B, layer: 1, pos: 2431
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2801
type: B, layer: 1, pos: 2801
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 3129
type: A, layer: 1, pos: 3129
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 3033
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 2627
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2829
type: B, layer: 1, pos: 2829
type: B, layer: 1, pos: 2573
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 2831
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 3318
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 3541
type: A, layer: 1, pos: 3541
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2850
type: B, layer: 1, pos: 2850
type: A, layer: 1, pos: 2088
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 319
type: A, layer: 1, pos: 3236
type: B, layer: 1, pos: 3236
type: A, layer: 1, pos: 319
type: A, layer: 1, pos: 2150
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 2776
type: A, layer: 1, pos: 2776
type: B, layer: 1, pos: 2150
type: A, layer: 1, pos: 2832
type: B, layer: 1, pos: 2832
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: A, layer: 1, pos: 192
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 3501
type: B, layer: 1, pos: 2656
type: A, layer: 1, pos: 3501
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 3266
type: B, layer: 1, pos: 3266
type: B, layer: 1, pos: 2805
type: A, layer: 1, pos: 2805
type: B, layer: 1, pos: 2891
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2507
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 2033
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2067
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 2055
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 2056
type: A, layer: 1, pos: 3426
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 501
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 374
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 2669
type: A, layer: 1, pos: 2819
type: B, layer: 1, pos: 374
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 2669
type: B, layer: 1, pos: 2819
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 2056
type: B, layer: 1, pos: 2071
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 3123

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 367

## Relational analysis of NS_A1_A2_A1

### Relational analysis result of NS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262014, upper bound: 0.0257186
time: 103.91 seconds

## Relational analysis of NS_A1_A2_A2

### Relational analysis result of NS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262005, upper bound: 0.0257190
time: 100.77 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -3.2361124, -2.5154257, -3.2361524, -2.5162485, -0.2139184, 0.2156757
1: -1.2226143, -0.3935633, -1.2226667, -0.3951278, -0.2470379, 0.2499986
2: -1.9692442, -1.6593912, -1.9680979, -1.6592722, -0.0577990, 0.0550920
3: -0.7501779, -0.4575962, -0.7469431, -0.4574821, -0.0776844, 0.0716148
4: -2.7781115, -2.2691815, -2.7781379, -2.2692986, -0.1081041, 0.1080868
5: -0.2424438, 0.0984244, -0.2394378, 0.0985482, -0.0744400, 0.0689634
6: -0.1061621, 0.2174038, -0.1062267, 0.2169081, -0.2155032, 0.2162530
7: -0.7135944, -0.3448977, -0.7136496, -0.3451009, -0.0632884, 0.0647632
8: -2.5323029, -1.6468301, -2.5324898, -1.6465330, -0.6461887, 0.6462727
9: -1.5337200, -0.6891165, -1.5336609, -0.6906610, -0.4284701, 0.4309857

Time for backsubstitution: 5.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 367
type: A, layer: 1, pos: 367
type: B, layer: 1, pos: 2350
type: A, layer: 1, pos: 2350
type: B, layer: 1, pos: 2152
type: A, layer: 1, pos: 2152
type: A, layer: 1, pos: 2351
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2405
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 3079
type: A, layer: 1, pos: 3079
type: B, layer: 1, pos: 2544
type: A, layer: 1, pos: 2544
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 2604
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 261
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2649
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 259
type: A, layer: 1, pos: 259
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 3441
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 3130
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 3130
type: B, layer: 1, pos: 2097
type: A, layer: 1, pos: 2097
type: B, layer: 1, pos: 2637
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 3496
type: A, layer: 1, pos: 3496
type: B, layer: 1, pos: 2431
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2801
type: B, layer: 1, pos: 2801
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 3129
type: A, layer: 1, pos: 3129
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 3033
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 2627
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2829
type: B, layer: 1, pos: 2829
type: B, layer: 1, pos: 2573
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 2831
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 3318
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 3541
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 2850
type: B, layer: 1, pos: 2850
type: A, layer: 1, pos: 2088
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 319
type: B, layer: 1, pos: 3236
type: A, layer: 1, pos: 3236
type: A, layer: 1, pos: 319
type: A, layer: 1, pos: 2150
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 2776
type: A, layer: 1, pos: 2776
type: B, layer: 1, pos: 2150
type: A, layer: 1, pos: 2832
type: B, layer: 1, pos: 2832
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: A, layer: 1, pos: 192
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 3501
type: B, layer: 1, pos: 2656
type: A, layer: 1, pos: 3501
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 3266
type: B, layer: 1, pos: 3266
type: B, layer: 1, pos: 2805
type: A, layer: 1, pos: 2805
type: B, layer: 1, pos: 2891
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2507
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2033
type: A, layer: 1, pos: 2033
type: B, layer: 1, pos: 2067
type: A, layer: 1, pos: 2067
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 2055
type: B, layer: 1, pos: 2055
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2056
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 917
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 925
type: B, layer: 1, pos: 910
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 901
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 909
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 928
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 374
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 2669
type: A, layer: 1, pos: 2819
type: B, layer: 1, pos: 374
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 2669
type: B, layer: 1, pos: 2819
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 2071
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 3123

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 367

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0260672, upper bound: 0.0260743
time: 87.77 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0260675, upper bound: 0.0260763
time: 21.92 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -3.2380781, -2.5153222, -3.2361593, -2.5162349, -0.2160696, 0.2158132
1: -1.2224231, -0.3934550, -1.2224989, -0.3951273, -0.2470981, 0.2500104
2: -1.9711506, -1.6591805, -1.9680984, -1.6591026, -0.0604852, 0.0551953
3: -0.7529073, -0.4573676, -0.7469807, -0.4573897, -0.0817650, 0.0719354
4: -2.7785313, -2.2693167, -2.7781410, -2.2694988, -0.1088366, 0.1082407
5: -0.2449840, 0.0985933, -0.2394417, 0.0986215, -0.0782573, 0.0691254
6: -0.1066554, 0.2178144, -0.1065470, 0.2169086, -0.2158949, 0.2169015
7: -0.7148480, -0.3452746, -0.7136528, -0.3453999, -0.0650884, 0.0647702
8: -2.5349507, -1.6464610, -2.5324883, -1.6463385, -0.6487808, 0.6463461
9: -1.5333285, -0.6889200, -1.5331230, -0.6906610, -0.4290202, 0.4306107

Time for backsubstitution: 5.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 367
type: A, layer: 1, pos: 367
type: B, layer: 1, pos: 2350
type: A, layer: 1, pos: 2350
type: B, layer: 1, pos: 2152
type: A, layer: 1, pos: 2152
type: A, layer: 1, pos: 2351
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2405
type: B, layer: 1, pos: 2405
type: A, layer: 1, pos: 3079
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2544
type: A, layer: 1, pos: 2544
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 2604
type: B, layer: 1, pos: 2604
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 261
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 784
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2649
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 259
type: A, layer: 1, pos: 259
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 3441
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: B, layer: 1, pos: 3130
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 3130
type: B, layer: 1, pos: 2097
type: A, layer: 1, pos: 2097
type: B, layer: 1, pos: 2637
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 3496
type: A, layer: 1, pos: 3496
type: B, layer: 1, pos: 2431
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 2801
type: B, layer: 1, pos: 2801
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 3129
type: A, layer: 1, pos: 3129
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 3033
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 2627
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2829
type: B, layer: 1, pos: 2829
type: B, layer: 1, pos: 2573
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 2831
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 3318
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 3541
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 2850
type: B, layer: 1, pos: 2850
type: A, layer: 1, pos: 2088
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 319
type: B, layer: 1, pos: 3236
type: A, layer: 1, pos: 3236
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 319
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 2776
type: A, layer: 1, pos: 2776
type: B, layer: 1, pos: 2150
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2832
type: B, layer: 1, pos: 2832
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: A, layer: 1, pos: 192
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 3501
type: A, layer: 1, pos: 3501
type: B, layer: 1, pos: 2656
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 3266
type: B, layer: 1, pos: 3266
type: B, layer: 1, pos: 2805
type: A, layer: 1, pos: 2805
type: B, layer: 1, pos: 2891
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 2507
type: A, layer: 1, pos: 2507
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 2033
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 2056
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 679
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 910
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 916
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 679
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 927
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 900
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 911
type: A, layer: 1, pos: 374
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 2669
type: A, layer: 1, pos: 2819
type: B, layer: 1, pos: 374
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 2669
type: B, layer: 1, pos: 2819
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 3426
type: B, layer: 1, pos: 2071
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 2442

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 367

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0260655, upper bound: 0.0262034
time: 133.70 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0260669, upper bound: 0.0262041
time: 43.37 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 182.68 seconds
NS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 182.68
Output dim: 3, lower bound: -0.0260673, upper bound: 0.0257240
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 182.68
Output dim: 3, lower bound: -0.0262031, upper bound: 0.0257267
NS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 182.68
Output dim: 3, lower bound: -0.0262014, upper bound: 0.0257186
NS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 182.68
Output dim: 3, lower bound: -0.0262005, upper bound: 0.0257190
NS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 182.68
Output dim: 3, lower bound: -0.0260672, upper bound: 0.0260743
NS_A2_A1_B2, status: Status.VERIFIED, split count: 3, time: 182.68
Output dim: 3, lower bound: -0.0260675, upper bound: 0.0260763
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 182.68
Output dim: 3, lower bound: -0.0260655, upper bound: 0.0262034
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 182.68
Output dim: 3, lower bound: -0.0260669, upper bound: 0.0262041

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: -3.2354918, -2.5165040, -3.2356424, -2.5164461, -0.2125614, 0.2138939
1: -1.2212310, -0.3951306, -1.2214456, -0.3951292, -0.2467242, 0.2469813
2: -1.9680283, -1.6601720, -1.9680176, -1.6599046, -0.0553160, 0.0512523
3: -0.7468485, -0.4599125, -0.7468731, -0.4593689, -0.0702988, 0.0711144
4: -2.7780414, -2.2693243, -2.7780590, -2.2693024, -0.1080934, 0.1032616
5: -0.2394289, 0.0960840, -0.2394315, 0.0966352, -0.0670859, 0.0685779
6: -0.1053579, 0.2167311, -0.1055951, 0.2167032, -0.2158025, 0.2131051
7: -0.7135727, -0.3451344, -0.7136051, -0.3451134, -0.0631777, 0.0579392
8: -2.5324886, -1.6464000, -2.5324905, -1.6461930, -0.6457839, 0.6465368
9: -1.5323119, -0.6906633, -1.5326295, -0.6906633, -0.4283454, 0.4254041

Time for backsubstitution: 5.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2350
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2152
type: B, layer: 1, pos: 2152
type: A, layer: 1, pos: 2351
type: B, layer: 1, pos: 2351
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2405
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 3079
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2544
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 2604
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2104
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 261
type: B, layer: 1, pos: 261
type: A, layer: 1, pos: 784
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 2403
type: A, layer: 1, pos: 2403
type: B, layer: 1, pos: 2649
type: A, layer: 1, pos: 2649
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 259
type: A, layer: 1, pos: 259
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 3441
type: B, layer: 1, pos: 3441
type: B, layer: 1, pos: 745
type: A, layer: 1, pos: 745
type: B, layer: 1, pos: 65
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 2634
type: B, layer: 1, pos: 2634
type: A, layer: 1, pos: 3216
type: B, layer: 1, pos: 3216
type: A, layer: 1, pos: 3130
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 2097
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2637
type: B, layer: 1, pos: 2637
type: A, layer: 1, pos: 3496
type: B, layer: 1, pos: 3496
type: A, layer: 1, pos: 2431
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 2801
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2828
type: B, layer: 1, pos: 2828
type: A, layer: 1, pos: 3129
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 3033
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 2627
type: B, layer: 1, pos: 2627
type: A, layer: 1, pos: 2829
type: B, layer: 1, pos: 2829
type: A, layer: 1, pos: 2573
type: B, layer: 1, pos: 2573
type: A, layer: 1, pos: 557
type: B, layer: 1, pos: 557
type: A, layer: 1, pos: 2831
type: B, layer: 1, pos: 2831
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 3318
type: A, layer: 1, pos: 3318
type: B, layer: 1, pos: 747
type: A, layer: 1, pos: 747
type: B, layer: 1, pos: 3541
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 2850
type: A, layer: 1, pos: 2850
type: A, layer: 1, pos: 2088
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 319
type: A, layer: 1, pos: 319
type: A, layer: 1, pos: 3236
type: B, layer: 1, pos: 3236
type: B, layer: 1, pos: 731
type: A, layer: 1, pos: 731
type: B, layer: 1, pos: 2150
type: A, layer: 1, pos: 2150
type: B, layer: 1, pos: 2776
type: A, layer: 1, pos: 2776
type: B, layer: 1, pos: 2832
type: A, layer: 1, pos: 2832
type: A, layer: 1, pos: 2314
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2656
type: A, layer: 1, pos: 2656
type: B, layer: 1, pos: 192
type: A, layer: 1, pos: 192
type: B, layer: 1, pos: 3501
type: A, layer: 1, pos: 3501
type: A, layer: 1, pos: 3266
type: B, layer: 1, pos: 3266
type: B, layer: 1, pos: 2891
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2805
type: B, layer: 1, pos: 2805
type: B, layer: 1, pos: 2889
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 2566
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2507
type: A, layer: 1, pos: 2507
type: B, layer: 1, pos: 738
type: A, layer: 1, pos: 738
type: B, layer: 1, pos: 2800
type: A, layer: 1, pos: 2800
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2442
type: B, layer: 1, pos: 781
type: A, layer: 1, pos: 781
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 2033
type: A, layer: 1, pos: 2033
type: B, layer: 1, pos: 2067
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 2067
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 3426
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 2055
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 529
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 2056
type: A, layer: 1, pos: 501
type: B, layer: 1, pos: 501
type: A, layer: 1, pos: 502
type: B, layer: 1, pos: 502
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2056
type: B, layer: 1, pos: 973
type: A, layer: 1, pos: 973
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 958
type: A, layer: 1, pos: 958
type: B, layer: 1, pos: 942
type: A, layer: 1, pos: 942
type: B, layer: 1, pos: 930
type: A, layer: 1, pos: 930
type: B, layer: 1, pos: 908
type: A, layer: 1, pos: 908
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 917
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 972
type: B, layer: 1, pos: 931
type: A, layer: 1, pos: 931
type: B, layer: 1, pos: 902
type: A, layer: 1, pos: 902
type: B, layer: 1, pos: 926
type: A, layer: 1, pos: 926
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 916
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 925
type: A, layer: 1, pos: 909
type: B, layer: 1, pos: 912
type: A, layer: 1, pos: 912
type: B, layer: 1, pos: 901
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 915
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 915
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 900
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 907
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 913
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 927
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 374
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 2669
type: A, layer: 1, pos: 2819
type: B, layer: 1, pos: 374
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 2669
type: B, layer: 1, pos: 2819
type: A, layer: 1, pos: 907
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 3123

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2350

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261964, upper bound: 0.0255860
time: 89.10 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261946, upper bound: 0.0257198
time: 9.48 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 188.10 + 1632.95 = 1821.05 seconds
