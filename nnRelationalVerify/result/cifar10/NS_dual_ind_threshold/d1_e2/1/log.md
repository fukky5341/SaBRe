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
execution time: IAR + RelationalAnalysis = 7.16 + 182.21 = 189.37 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0262224, upper bound: 0.0262229

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 385
type: A, layer: 1, pos: 378
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2152
type: A, layer: 1, pos: 2351
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3501
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 259
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 3496
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2829
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 3441
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 3318
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2850
type: A, layer: 1, pos: 2832
type: A, layer: 1, pos: 192
type: A, layer: 1, pos: 3236
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 319
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 3266
type: A, layer: 1, pos: 2805
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 374
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 2669
type: A, layer: 1, pos: 2819

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 385

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262193, upper bound: 0.0258713
time: 22.79 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262196, upper bound: 0.0262210
time: 72.52 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 95.37 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 95.37
Output dim: 3, lower bound: -0.0262193, upper bound: 0.0258713
NS_A2, status: Status.UNKNOWN, split count: 1, time: 95.37
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

Time for backsubstitution: 5.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2152
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3501
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 259
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 3496
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2829
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 3441
type: B, layer: 1, pos: 2831
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 3541
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2850
type: B, layer: 1, pos: 2832
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 3236
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 319
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 3266
type: B, layer: 1, pos: 2805
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 374
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 2669
type: B, layer: 1, pos: 2819

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 378

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0260772, upper bound: 0.0258571
time: 134.20 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262054, upper bound: 0.0258572
time: 14.45 seconds

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

Time for backsubstitution: 5.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 378
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2152
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3501
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 259
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 3496
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2829
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 3441
type: B, layer: 1, pos: 2831
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 3541
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2850
type: B, layer: 1, pos: 2832
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 3236
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 319
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 3266
type: B, layer: 1, pos: 2805
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 374
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 2669
type: B, layer: 1, pos: 2819

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 378

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0260761, upper bound: 0.0258576
time: 47.84 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262052, upper bound: 0.0262067
time: 102.02 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 155.40 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 155.40
Output dim: 3, lower bound: -0.0260772, upper bound: 0.0258571
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 155.40
Output dim: 3, lower bound: -0.0262054, upper bound: 0.0258572
NS_A2_B1, status: Status.VERIFIED, split count: 2, time: 155.40
Output dim: 3, lower bound: -0.0260761, upper bound: 0.0258576
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 155.40
Output dim: 3, lower bound: -0.0262052, upper bound: 0.0262067

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -3.2355058, -2.5162487, -3.2376099, -2.5162487, -0.2139061, 0.2161757
1: -1.2213049, -0.3951287, -1.2215204, -0.3950214, -0.2470406, 0.2473224
2: -1.9680948, -1.6599104, -1.9700013, -1.6597855, -0.0553039, 0.0578170
3: -0.7469017, -0.4596971, -0.7495871, -0.4592330, -0.0720554, 0.0754817
4: -2.7781558, -2.2695029, -2.7786026, -2.2694573, -0.1082231, 0.1088490
5: -0.2394344, 0.0962902, -0.2419704, 0.0967025, -0.0693151, 0.0726800
6: -0.1057926, 0.2169067, -0.1059778, 0.2173167, -0.2156376, 0.2153947
7: -0.7135993, -0.3454142, -0.7148840, -0.3455088, -0.0631907, 0.0650866
8: -2.5324905, -1.6458941, -2.5351381, -1.6458521, -0.6467481, 0.6493759
9: -1.5318503, -0.6906624, -1.5322104, -0.6904640, -0.4280534, 0.4292121

Time for backsubstitution: 5.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2152
type: A, layer: 1, pos: 2351
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3501
type: A, layer: 1, pos: 378
type: A, layer: 1, pos: 259
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 3496
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2829
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 3441
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 3318
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2850
type: A, layer: 1, pos: 2832
type: A, layer: 1, pos: 192
type: A, layer: 1, pos: 3236
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 319
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 3266
type: A, layer: 1, pos: 2805
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 374
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 2669
type: A, layer: 1, pos: 2819

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 367

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262008, upper bound: 0.0257179
time: 311.62 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262011, upper bound: 0.0257198
time: 7.61 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -3.2361236, -2.5153210, -3.2381139, -2.5162361, -0.2139492, 0.2179337
1: -1.2224607, -0.3935633, -1.2224603, -0.3950195, -0.2470596, 0.2500488
2: -1.9692454, -1.6591301, -1.9700043, -1.6591535, -0.0577766, 0.0579041
3: -0.7502282, -0.4573946, -0.7496541, -0.4573629, -0.0778741, 0.0758246
4: -2.7781155, -2.2693610, -2.7785578, -2.2694535, -0.1082313, 0.1088465
5: -0.2424491, 0.0986165, -0.2419764, 0.0985981, -0.0744440, 0.0729384
6: -0.1065954, 0.2174044, -0.1066072, 0.2173184, -0.2162639, 0.2165323
7: -0.7135988, -0.3451775, -0.7149015, -0.3454970, -0.0632604, 0.0666000
8: -2.5323038, -1.6464539, -2.5351367, -1.6463461, -0.6461062, 0.6490195
9: -1.5332375, -0.6891165, -1.5332155, -0.6904631, -0.4281411, 0.4314909

Time for backsubstitution: 5.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2152
type: A, layer: 1, pos: 2351
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3501
type: A, layer: 1, pos: 378
type: A, layer: 1, pos: 259
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 3496
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2829
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 3441
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 3318
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2850
type: A, layer: 1, pos: 2832
type: A, layer: 1, pos: 3236
type: A, layer: 1, pos: 192
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 319
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 3266
type: A, layer: 1, pos: 2805
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 374
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 2669
type: A, layer: 1, pos: 2819

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 367

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262003, upper bound: 0.0260694
time: 72.44 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0262025, upper bound: 0.0262018
time: 77.78 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 155.81 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 155.81
Output dim: 3, lower bound: -0.0262008, upper bound: 0.0257179
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 155.81
Output dim: 3, lower bound: -0.0262011, upper bound: 0.0257198
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 155.81
Output dim: 3, lower bound: -0.0262003, upper bound: 0.0260694
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 155.81
Output dim: 3, lower bound: -0.0262025, upper bound: 0.0262018

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.2351747, -2.5168102, -3.2375169, -2.5167251, -0.2125762, 0.2153201
1: -1.2208462, -0.3954978, -1.2211447, -0.3950362, -0.2465435, 0.2468421
2: -1.9659932, -1.6615841, -1.9682057, -1.6598525, -0.0530583, 0.0540038
3: -0.7457614, -0.4609370, -0.7495652, -0.4602705, -0.0704262, 0.0744597
4: -2.7753921, -2.2715554, -2.7762918, -2.2694817, -0.1050740, 0.1037580
5: -0.2379678, 0.0946736, -0.2419678, 0.0953396, -0.0671096, 0.0714321
6: -0.1047792, 0.2166338, -0.1056201, 0.2170922, -0.2144889, 0.2134246
7: -0.7100582, -0.3480843, -0.7119547, -0.3455437, -0.0596942, 0.0595584
8: -2.5321131, -1.6464977, -2.5351238, -1.6463671, -0.6457171, 0.6486895
9: -1.5296268, -0.6925445, -1.5303049, -0.6904855, -0.4261544, 0.4258747

Time for backsubstitution: 5.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2152
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3501
type: B, layer: 1, pos: 259
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 3496
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2829
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 3441
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 2831
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 3541
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2850
type: B, layer: 1, pos: 2832
type: B, layer: 1, pos: 3236
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 319
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 3266
type: B, layer: 1, pos: 2805
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 907
type: B, layer: 1, pos: 374
type: B, layer: 1, pos: 914
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 944
type: B, layer: 1, pos: 959
type: B, layer: 1, pos: 974
type: B, layer: 1, pos: 2669
type: B, layer: 1, pos: 2819

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2350

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261943, upper bound: 0.0255758
time: 55.37 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261940, upper bound: 0.0257124
time: 65.38 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.2355001, -2.5164342, -3.2376049, -2.5163999, -0.2138806, 0.2148585
1: -1.2210250, -0.3951287, -1.2212920, -0.3950210, -0.2467496, 0.2470279
2: -1.9680172, -1.6599112, -1.9699359, -1.6597860, -0.0515513, 0.0578066
3: -0.7468991, -0.4597138, -0.7495850, -0.4592467, -0.0718997, 0.0739167
4: -2.7780299, -2.2695036, -2.7784920, -2.2694576, -0.1033950, 0.1088463
5: -0.2394344, 0.0962728, -0.2419704, 0.0966883, -0.0691881, 0.0704550
6: -0.1057900, 0.2167034, -0.1059754, 0.2171417, -0.2136272, 0.2163183
7: -0.7135720, -0.3454146, -0.7148616, -0.3455092, -0.0578790, 0.0650454
8: -2.5324898, -1.6460452, -2.5351381, -1.6459832, -0.6466570, 0.6483278
9: -1.5318251, -0.6906624, -1.5321908, -0.6904650, -0.4247522, 0.4291756

Time for backsubstitution: 5.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2152
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3501
type: B, layer: 1, pos: 259
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 3496
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2829
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 3441
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 2831
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 3541
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2850
type: B, layer: 1, pos: 2832
type: B, layer: 1, pos: 3236
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 319
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 3266
type: B, layer: 1, pos: 2805
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 907
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
type: B, layer: 1, pos: 2350

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261970, upper bound: 0.0255784
time: 82.12 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261961, upper bound: 0.0258488
time: 7.79 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.2357907, -2.5158815, -3.2380214, -2.5167131, -0.2126190, 0.2170779
1: -1.2220001, -0.3939323, -1.2220840, -0.3950338, -0.2465626, 0.2495681
2: -1.9671439, -1.6608040, -1.9682086, -1.6592209, -0.0555309, 0.0540909
3: -0.7490887, -0.4586347, -0.7496324, -0.4584003, -0.0762464, 0.0748027
4: -2.7753503, -2.2714128, -2.7762473, -2.2694781, -0.1050820, 0.1037554
5: -0.2409824, 0.0969999, -0.2419739, 0.0972350, -0.0722387, 0.0716906
6: -0.1055801, 0.2171313, -0.1062500, 0.2170938, -0.2151141, 0.2145625
7: -0.7100573, -0.3478476, -0.7119727, -0.3455319, -0.0597640, 0.0610719
8: -2.5319266, -1.6470585, -2.5351222, -1.6468601, -0.6450748, 0.6483335
9: -1.5310159, -0.6909990, -1.5313106, -0.6904850, -0.4262416, 0.4281533

Time for backsubstitution: 5.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2152
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3501
type: B, layer: 1, pos: 259
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 3496
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2829
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 3441
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 2831
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 3541
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2850
type: B, layer: 1, pos: 2832
type: B, layer: 1, pos: 3236
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 319
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 3266
type: B, layer: 1, pos: 2805
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 907
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
type: B, layer: 1, pos: 2350

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261944, upper bound: 0.0259268
time: 64.50 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261943, upper bound: 0.0260631
time: 8.16 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.2361169, -2.5155058, -3.2381091, -2.5163870, -0.2139238, 0.2166163
1: -1.2221799, -0.3935637, -1.2222314, -0.3950195, -0.2467684, 0.2497542
2: -1.9691672, -1.6591308, -1.9699388, -1.6591544, -0.0540241, 0.0578937
3: -0.7502258, -0.4574113, -0.7496520, -0.4573762, -0.0777182, 0.0742598
4: -2.7779896, -2.2693615, -2.7784467, -2.2694540, -0.1034033, 0.1088437
5: -0.2424490, 0.0985991, -0.2419762, 0.0985839, -0.0743170, 0.0707134
6: -0.1065927, 0.2172010, -0.1066052, 0.2171433, -0.2142538, 0.2174568
7: -0.7135713, -0.3451779, -0.7148795, -0.3454974, -0.0579485, 0.0665589
8: -2.5323019, -1.6466060, -2.5351362, -1.6464767, -0.6460142, 0.6479709
9: -1.5332136, -0.6891179, -1.5331950, -0.6904645, -0.4248393, 0.4314544

Time for backsubstitution: 5.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2350
type: B, layer: 1, pos: 2152
type: B, layer: 1, pos: 2351
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2405
type: B, layer: 1, pos: 3079
type: B, layer: 1, pos: 2544
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 261
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2104
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 385
type: B, layer: 1, pos: 2649
type: B, layer: 1, pos: 2403
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 3501
type: B, layer: 1, pos: 259
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 3130
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 2634
type: B, layer: 1, pos: 3129
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 3216
type: B, layer: 1, pos: 745
type: B, layer: 1, pos: 65
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 2097
type: B, layer: 1, pos: 2801
type: B, layer: 1, pos: 2828
type: B, layer: 1, pos: 3496
type: B, layer: 1, pos: 2627
type: B, layer: 1, pos: 2573
type: B, layer: 1, pos: 2431
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 3033
type: B, layer: 1, pos: 2829
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 3441
type: B, layer: 1, pos: 557
type: B, layer: 1, pos: 2831
type: B, layer: 1, pos: 2618
type: B, layer: 1, pos: 3318
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 2150
type: B, layer: 1, pos: 726
type: B, layer: 1, pos: 2088
type: B, layer: 1, pos: 3541
type: B, layer: 1, pos: 3123
type: B, layer: 1, pos: 2314
type: B, layer: 1, pos: 2850
type: B, layer: 1, pos: 2832
type: B, layer: 1, pos: 3236
type: B, layer: 1, pos: 192
type: B, layer: 1, pos: 2889
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 319
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 2656
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 502
type: B, layer: 1, pos: 3426
type: B, layer: 1, pos: 3266
type: B, layer: 1, pos: 2805
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 2800
type: B, layer: 1, pos: 738
type: B, layer: 1, pos: 2507
type: B, layer: 1, pos: 501
type: B, layer: 1, pos: 2891
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 2067
type: B, layer: 1, pos: 781
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 529
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2055
type: B, layer: 1, pos: 2056
type: B, layer: 1, pos: 973
type: B, layer: 1, pos: 679
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 958
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 942
type: B, layer: 1, pos: 930
type: B, layer: 1, pos: 917
type: B, layer: 1, pos: 902
type: B, layer: 1, pos: 972
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 908
type: B, layer: 1, pos: 931
type: B, layer: 1, pos: 912
type: B, layer: 1, pos: 916
type: B, layer: 1, pos: 928
type: B, layer: 1, pos: 927
type: B, layer: 1, pos: 913
type: B, layer: 1, pos: 911
type: B, layer: 1, pos: 901
type: B, layer: 1, pos: 910
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 900
type: B, layer: 1, pos: 925
type: B, layer: 1, pos: 926
type: B, layer: 1, pos: 909
type: B, layer: 1, pos: 907
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
type: B, layer: 1, pos: 2350

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261974, upper bound: 0.0259254
time: 179.88 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261958, upper bound: 0.0260621
time: 143.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 329.15 seconds
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 329.15
Output dim: 3, lower bound: -0.0261943, upper bound: 0.0255758
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 329.15
Output dim: 3, lower bound: -0.0261940, upper bound: 0.0257124
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 329.15
Output dim: 3, lower bound: -0.0261970, upper bound: 0.0255784
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 329.15
Output dim: 3, lower bound: -0.0261961, upper bound: 0.0258488
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 329.15
Output dim: 3, lower bound: -0.0261944, upper bound: 0.0259268
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 329.15
Output dim: 3, lower bound: -0.0261943, upper bound: 0.0260631
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 329.15
Output dim: 3, lower bound: -0.0261974, upper bound: 0.0259254
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 329.15
Output dim: 3, lower bound: -0.0261958, upper bound: 0.0260621

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -3.2351685, -2.5205951, -3.2329547, -2.5213547, -0.2086073, 0.2080763
1: -1.2208390, -0.3987927, -1.2171774, -0.3990722, -0.2430270, 0.2404056
2: -1.9659355, -1.6615843, -1.9681182, -1.6598890, -0.0528991, 0.0539068
3: -0.7449398, -0.4609370, -0.7485709, -0.4614131, -0.0683650, 0.0733350
4: -2.7753785, -2.2715626, -2.7762644, -2.2694926, -0.1049006, 0.1036789
5: -0.2368225, 0.0946690, -0.2405903, 0.0939342, -0.0647524, 0.0701464
6: -0.1038070, 0.2166338, -0.1043450, 0.2159836, -0.2116065, 0.2117959
7: -0.7100571, -0.3480860, -0.7119461, -0.3455470, -0.0596786, 0.0595382
8: -2.5320995, -1.6490335, -2.5321012, -1.6494570, -0.6427145, 0.6432810
9: -1.5295992, -0.6942525, -1.5282702, -0.6925654, -0.4242430, 0.4224446

Time for backsubstitution: 5.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2152
type: A, layer: 1, pos: 2351
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3501
type: A, layer: 1, pos: 378
type: A, layer: 1, pos: 259
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 3496
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2829
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 3441
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 3318
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2850
type: A, layer: 1, pos: 2832
type: A, layer: 1, pos: 192
type: A, layer: 1, pos: 3236
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 319
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 3266
type: A, layer: 1, pos: 2805
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 374
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 2669
type: A, layer: 1, pos: 2819

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2152

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0261309, upper bound: 0.0255623
time: 37.99 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261542, upper bound: 0.0255621
time: 35.47 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -3.2351744, -2.5168204, -3.2375159, -2.5167372, -0.2052672, 0.2152434
1: -1.2208447, -0.3955064, -1.2211442, -0.3950467, -0.2402381, 0.2467996
2: -1.9659724, -1.6615844, -1.9681804, -1.6598524, -0.0530578, 0.0538934
3: -0.7457030, -0.4609370, -0.7495054, -0.4602704, -0.0704180, 0.0722850
4: -2.7753916, -2.2715664, -2.7762918, -2.2694955, -0.1050687, 0.1035942
5: -0.2379616, 0.0946733, -0.2419607, 0.0953392, -0.0671048, 0.0690009
6: -0.1044978, 0.2166337, -0.1052946, 0.2170922, -0.2144803, 0.2114483
7: -0.7100545, -0.3480845, -0.7119501, -0.3455438, -0.0596840, 0.0595422
8: -2.5321124, -1.6465077, -2.5351229, -1.6463790, -0.6425176, 0.6486115
9: -1.5296254, -0.6925530, -1.5303025, -0.6904969, -0.4239721, 0.4258196

Time for backsubstitution: 5.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2152
type: A, layer: 1, pos: 2351
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2405
type: A, layer: 1, pos: 3079
type: A, layer: 1, pos: 2544
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 261
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2104
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2649
type: A, layer: 1, pos: 2403
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 3501
type: A, layer: 1, pos: 378
type: A, layer: 1, pos: 259
type: A, layer: 1, pos: 3130
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 2634
type: A, layer: 1, pos: 3129
type: A, layer: 1, pos: 3216
type: A, layer: 1, pos: 745
type: A, layer: 1, pos: 65
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 2097
type: A, layer: 1, pos: 2350
type: A, layer: 1, pos: 2801
type: A, layer: 1, pos: 2828
type: A, layer: 1, pos: 3496
type: A, layer: 1, pos: 2627
type: A, layer: 1, pos: 2573
type: A, layer: 1, pos: 2431
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 2829
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 3441
type: A, layer: 1, pos: 2831
type: A, layer: 1, pos: 557
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 3318
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 2150
type: A, layer: 1, pos: 726
type: A, layer: 1, pos: 2088
type: A, layer: 1, pos: 3541
type: A, layer: 1, pos: 3123
type: A, layer: 1, pos: 2314
type: A, layer: 1, pos: 2850
type: A, layer: 1, pos: 2832
type: A, layer: 1, pos: 192
type: A, layer: 1, pos: 3236
type: A, layer: 1, pos: 2889
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 319
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 2656
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 502
type: A, layer: 1, pos: 3426
type: A, layer: 1, pos: 3266
type: A, layer: 1, pos: 2805
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 2800
type: A, layer: 1, pos: 738
type: A, layer: 1, pos: 2507
type: A, layer: 1, pos: 501
type: A, layer: 1, pos: 2891
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 2067
type: A, layer: 1, pos: 781
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 529
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2055
type: A, layer: 1, pos: 2056
type: A, layer: 1, pos: 973
type: A, layer: 1, pos: 679
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 958
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 942
type: A, layer: 1, pos: 930
type: A, layer: 1, pos: 917
type: A, layer: 1, pos: 902
type: A, layer: 1, pos: 972
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 908
type: A, layer: 1, pos: 931
type: A, layer: 1, pos: 912
type: A, layer: 1, pos: 916
type: A, layer: 1, pos: 928
type: A, layer: 1, pos: 927
type: A, layer: 1, pos: 913
type: A, layer: 1, pos: 911
type: A, layer: 1, pos: 901
type: A, layer: 1, pos: 910
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 900
type: A, layer: 1, pos: 925
type: A, layer: 1, pos: 909
type: A, layer: 1, pos: 926
type: A, layer: 1, pos: 907
type: A, layer: 1, pos: 374
type: A, layer: 1, pos: 914
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 944
type: A, layer: 1, pos: 959
type: A, layer: 1, pos: 974
type: A, layer: 1, pos: 2669
type: A, layer: 1, pos: 2819

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2152

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0261296, upper bound: 0.0256963
time: 98.33 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0261542, upper bound: 0.0256971
time: 33.02 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 189.37 + 1730.82 = 1920.19 seconds
