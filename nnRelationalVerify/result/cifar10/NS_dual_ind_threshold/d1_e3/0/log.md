## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 0)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0146240613


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317528, 0.0317528)
1: (-1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3185107, 0.3185106)
2: (-0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496899, 0.0496899)
3: (-1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0872872, 0.0872872)
4: (-1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611762, 0.1611763)
5: (-1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0887314, 0.0887314)
6: (-5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1424298, 0.1424298)
7: (-2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1352779, 0.1352779)
8: (-0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314896, 0.0314896)
9: (-2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1832750, 0.1832750)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.63 + 24.48 = 32.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0146387, upper bound: 0.0146385

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3567
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 3206
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 3102
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2027
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2810
type: A, layer: 1, pos: 2098
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2645
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2879

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 3567

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0145022, upper bound: 0.0146385
time: 4.23 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146385, upper bound: 0.0146385
time: 2.70 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.99 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.99
Output dim: 0, lower bound: -0.0145022, upper bound: 0.0146385
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.99
Output dim: 0, lower bound: -0.0146385, upper bound: 0.0146385

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.9214237, 1.2475445, 0.9212788, 1.2475448, -0.0309486, 0.0310930
1: -1.7828455, -0.8735240, -1.7828460, -0.8734075, -0.3179811, 0.3178852
2: -0.6261997, -0.1499572, -0.6263443, -0.1499570, -0.0489208, 0.0490509
3: -1.4627888, -0.9429540, -1.4627887, -0.9429110, -0.0870883, 0.0870449
4: -1.8161993, -0.9508905, -1.8163227, -0.9508911, -0.1605852, 0.1606469
5: -1.9003955, -1.3566617, -1.9003959, -1.3566206, -0.0885420, 0.0885006
6: -5.1299977, -4.2965784, -5.1299977, -4.2964950, -0.1421208, 0.1420784
7: -2.9591422, -2.1592262, -2.9591417, -2.1591952, -0.1351053, 0.1350686
8: -0.7260492, -0.3993839, -0.7260493, -0.3992747, -0.0310078, 0.0309125
9: -2.6081171, -1.6745629, -2.6081169, -1.6743445, -0.1822835, 0.1820897

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 3206
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 3102
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2027
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2810
type: B, layer: 1, pos: 2098
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 302
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2029
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2879

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 335

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145020, upper bound: 0.0145890
time: 397.35 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0145020, upper bound: 0.0146383
time: 8.79 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.9206194, 1.2502403, 0.9206195, 1.2475445, -0.0310660, 0.0344184
1: -1.7860234, -0.8731272, -1.7828481, -0.8731041, -0.3214819, 0.3180024
2: -0.6270069, -0.1472657, -0.6270009, -0.1499570, -0.0490515, 0.0523562
3: -1.4635425, -0.9427196, -1.4627891, -0.9427193, -0.0879467, 0.0870795
4: -1.8170646, -0.9483186, -1.8168337, -0.9508910, -0.1615860, 0.1637030
5: -1.9011126, -1.3564342, -1.9003961, -1.3564341, -0.0893507, 0.0885344
6: -5.1319718, -4.2950149, -5.1299977, -4.2961540, -0.1443464, 0.1439553
7: -2.9596512, -2.1590669, -2.9591417, -2.1590722, -0.1357827, 0.1351017
8: -0.7281013, -0.3987780, -0.7260501, -0.3987791, -0.0335185, 0.0310031
9: -2.6125102, -1.6736748, -2.6081164, -1.6736639, -0.1874801, 0.1822823

Time for backsubstitution: 5.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 335
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 3206
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 3102
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2027
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2810
type: B, layer: 1, pos: 2098
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 302
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2029
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2879

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 335

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146383, upper bound: 0.0145890
time: 9.00 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146383, upper bound: 0.0146384
time: 5.25 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 20.31 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 20.31
Output dim: 0, lower bound: -0.0145020, upper bound: 0.0145890
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 20.31
Output dim: 0, lower bound: -0.0145020, upper bound: 0.0146383
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 20.31
Output dim: 0, lower bound: -0.0146383, upper bound: 0.0145890
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 20.31
Output dim: 0, lower bound: -0.0146383, upper bound: 0.0146384

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 0.9214299, 1.2475445, 0.9212857, 1.2475445, -0.0309435, 0.0298947
1: -1.7828453, -0.8735237, -1.7828457, -0.8734075, -0.3179772, 0.3178809
2: -0.6261961, -0.1500217, -0.6263399, -0.1500271, -0.0485958, 0.0490135
3: -1.4627515, -0.9429575, -1.4627457, -0.9429147, -0.0870468, 0.0851866
4: -1.8161876, -0.9508905, -1.8163083, -0.9508911, -0.1605706, 0.1566276
5: -1.9003389, -1.3566620, -1.9003298, -1.3566210, -0.0885201, 0.0869523
6: -5.1299391, -4.2965813, -5.1299305, -4.2964973, -0.1421130, 0.1376044
7: -2.9590967, -2.1592259, -2.9590898, -2.1591957, -0.1350964, 0.1322396
8: -0.7260487, -0.3994470, -0.7260493, -0.3993477, -0.0302338, 0.0309114
9: -2.6081166, -1.6746118, -2.6081171, -1.6743987, -0.1813773, 0.1820279

Time for backsubstitution: 5.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 3206
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 3102
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2027
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2810
type: A, layer: 1, pos: 2098
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2645
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2879

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2295

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0144936, upper bound: 0.0146153
time: 3.25 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0144936, upper bound: 0.0146298
time: 8.44 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.9213790, 1.2502401, 0.9214697, 1.2471297, -0.0298053, 0.0335159
1: -1.7860165, -0.8731285, -1.7827523, -0.8730986, -0.3214676, 0.3179013
2: -0.6269264, -0.1473709, -0.6269363, -0.1500787, -0.0486614, 0.0518853
3: -1.4623470, -0.9427853, -1.4614143, -0.9435103, -0.0859643, 0.0858013
4: -1.8144574, -0.9483186, -1.8137612, -0.9526476, -0.1574083, 0.1609165
5: -1.9001286, -1.3564383, -1.8992625, -1.3570235, -0.0878079, 0.0876054
6: -5.1294913, -4.2950945, -5.1271558, -4.2983003, -0.1394499, 0.1410414
7: -2.9579117, -2.1590669, -2.9571412, -2.1602080, -0.1329824, 0.1333570
8: -0.7280983, -0.3990723, -0.7259787, -0.3991033, -0.0330296, 0.0302604
9: -2.6125107, -1.6746364, -2.6077645, -1.6747944, -0.1863822, 0.1809246

Time for backsubstitution: 5.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 3206
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 3102
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2027
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2810
type: A, layer: 1, pos: 2098
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2645
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2879

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2295

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0145660
time: 5.01 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0145805
time: 122.43 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.9206254, 1.2502403, 0.9206264, 1.2475446, -0.0310609, 0.0332201
1: -1.7860236, -0.8731275, -1.7828476, -0.8731043, -0.3214784, 0.3179982
2: -0.6270031, -0.1473305, -0.6269966, -0.1500269, -0.0487265, 0.0523188
3: -1.4635049, -0.9427232, -1.4627459, -0.9427229, -0.0879051, 0.0852212
4: -1.8170521, -0.9483186, -1.8168194, -0.9508914, -0.1615716, 0.1596829
5: -1.9010550, -1.3564351, -1.9003308, -1.3564347, -0.0893287, 0.0869861
6: -5.1319137, -4.2950168, -5.1299305, -4.2961569, -0.1443385, 0.1394815
7: -2.9596062, -2.1590669, -2.9590905, -2.1590724, -0.1357737, 0.1322728
8: -0.7281009, -0.3988411, -0.7260498, -0.3988522, -0.0327445, 0.0310020
9: -2.6125104, -1.6737223, -2.6081164, -1.6737173, -0.1865739, 0.1822208

Time for backsubstitution: 5.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2295
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 3206
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 3102
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2027
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2810
type: A, layer: 1, pos: 2098
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2645
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2879

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2295

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146300, upper bound: 0.0146151
time: 19.79 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0146299
time: 7.90 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 33.68 seconds
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 33.68
Output dim: 0, lower bound: -0.0144936, upper bound: 0.0146153
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 33.68
Output dim: 0, lower bound: -0.0144936, upper bound: 0.0146298
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 33.68
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0145660
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 33.68
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0145805
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 33.68
Output dim: 0, lower bound: -0.0146300, upper bound: 0.0146151
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 33.68
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0146299

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.9214296, 1.2474378, 0.9212856, 1.2474482, -0.0309430, 0.0289053
1: -1.7818441, -0.8735240, -1.7819645, -0.8734070, -0.3133260, 0.3178296
2: -0.6261956, -0.1500583, -0.6263403, -0.1500598, -0.0485955, 0.0485875
3: -1.4627481, -0.9429578, -1.4627428, -0.9429145, -0.0869513, 0.0851864
4: -1.8161831, -0.9513472, -1.8163041, -0.9512879, -0.1605689, 0.1550683
5: -1.9003386, -1.3566949, -1.9003294, -1.3566499, -0.0885196, 0.0868067
6: -5.1297998, -4.2965832, -5.1298084, -4.2964988, -0.1416446, 0.1376039
7: -2.9590967, -2.1594830, -2.9590898, -2.1594200, -0.1350957, 0.1309577
8: -0.7260489, -0.3994498, -0.7260494, -0.3993501, -0.0302335, 0.0308781
9: -2.6079848, -1.6746111, -2.6080017, -1.6743982, -0.1807742, 0.1820274

Time for backsubstitution: 5.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 3206
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 3102
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2027
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2810
type: B, layer: 1, pos: 2098
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 302
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2029
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2879

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3472

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0144320, upper bound: 0.0146298
time: 4.44 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0144936, upper bound: 0.0146300
time: 6.64 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.9215217, 1.2498423, 0.9214701, 1.2467769, -0.0288710, 0.0329300
1: -1.7843361, -0.8729308, -1.7812855, -0.8730984, -0.3186111, 0.3133718
2: -0.6268304, -0.1475653, -0.6269358, -0.1502484, -0.0482543, 0.0516262
3: -1.4622892, -0.9428246, -1.4613633, -0.9435107, -0.0859008, 0.0857066
4: -1.8146906, -0.9488313, -1.8137560, -0.9530981, -0.1560079, 0.1597893
5: -1.9001364, -1.3564878, -1.8992624, -1.3570671, -0.0876712, 0.0875158
6: -5.1292777, -4.2950587, -5.1269689, -4.2983027, -0.1390473, 0.1406309
7: -2.9579701, -2.1594601, -2.9571412, -2.1605523, -0.1317892, 0.1326183
8: -0.7280892, -0.3990980, -0.7259786, -0.3991261, -0.0329981, 0.0302295
9: -2.6123085, -1.6746147, -2.6075871, -1.6747956, -0.1860038, 0.1803514

Time for backsubstitution: 5.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 3206
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 3102
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2027
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2810
type: B, layer: 1, pos: 2098
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 302
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2029
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2879

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3472

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145682, upper bound: 0.0145657
time: 3.55 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0145657
time: 173.17 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.9213789, 1.2501338, 0.9214696, 1.2470332, -0.0298048, 0.0325263
1: -1.7850132, -0.8731291, -1.7818723, -0.8730989, -0.3168176, 0.3178509
2: -0.6269261, -0.1474075, -0.6269363, -0.1501109, -0.0486611, 0.0514589
3: -1.4623435, -0.9427856, -1.4614115, -0.9435103, -0.0858687, 0.0858012
4: -1.8144523, -0.9487743, -1.8137579, -0.9530448, -0.1574064, 0.1593571
5: -1.9001284, -1.3564709, -1.8992624, -1.3570521, -0.0878076, 0.0874595
6: -5.1293507, -4.2950959, -5.1270332, -4.2983017, -0.1389810, 0.1410407
7: -2.9579117, -2.1593249, -2.9571412, -2.1604326, -0.1329816, 0.1320752
8: -0.7280981, -0.3990750, -0.7259786, -0.3991057, -0.0330293, 0.0302270
9: -2.6123786, -1.6746364, -2.6076491, -1.6747942, -0.1857785, 0.1809237

Time for backsubstitution: 5.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 3206
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 3102
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2027
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2810
type: B, layer: 1, pos: 2098
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 302
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2029
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2879

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3472

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145682, upper bound: 0.0145808
time: 59.90 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0145807
time: 5.91 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.9207685, 1.2498425, 0.9206266, 1.2471918, -0.0301268, 0.0326342
1: -1.7843430, -0.8729291, -1.7813811, -0.8731039, -0.3186220, 0.3134683
2: -0.6269073, -0.1475247, -0.6269956, -0.1501973, -0.0483197, 0.0520598
3: -1.4634475, -0.9427621, -1.4626949, -0.9427229, -0.0878416, 0.0851264
4: -1.8172859, -0.9488312, -1.8168142, -0.9513416, -0.1601680, 0.1585556
5: -1.9010623, -1.3564844, -1.9003307, -1.3564781, -0.0891919, 0.0868964
6: -5.1317005, -4.2949805, -5.1297445, -4.2961602, -0.1439360, 0.1390709
7: -2.9596643, -2.1594601, -2.9590900, -2.1594160, -0.1345805, 0.1315341
8: -0.7280919, -0.3988670, -0.7260499, -0.3988748, -0.0327130, 0.0309711
9: -2.6123078, -1.6737010, -2.6079402, -1.6737192, -0.1861958, 0.1816475

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 3206
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 3102
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2027
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2810
type: B, layer: 1, pos: 2098
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 302
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2029
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2879

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3472

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145682, upper bound: 0.0146151
time: 6.83 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0146152
time: 3.72 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.9206256, 1.2501335, 0.9206264, 1.2474480, -0.0310604, 0.0322308
1: -1.7850211, -0.8731272, -1.7819657, -0.8731041, -0.3168278, 0.3179467
2: -0.6270033, -0.1473669, -0.6269965, -0.1500598, -0.0487262, 0.0518927
3: -1.4635017, -0.9427232, -1.4627428, -0.9427229, -0.0878096, 0.0852211
4: -1.8170481, -0.9487742, -1.8168167, -0.9512883, -0.1615699, 0.1581230
5: -1.9010552, -1.3564680, -1.9003304, -1.3564634, -0.0893283, 0.0868402
6: -5.1317744, -4.2950187, -5.1298089, -4.2961593, -0.1438699, 0.1394807
7: -2.9596062, -2.1593244, -2.9590905, -2.1592970, -0.1357729, 0.1309910
8: -0.7281009, -0.3988439, -0.7260497, -0.3988543, -0.0327442, 0.0309687
9: -2.6123779, -1.6737230, -2.6080015, -1.6737180, -0.1859707, 0.1822200

Time for backsubstitution: 5.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2653
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 2305
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 3206
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2041
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2040
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2026
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 707
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 3102
type: B, layer: 1, pos: 2112
type: B, layer: 1, pos: 2027
type: B, layer: 1, pos: 2155
type: B, layer: 1, pos: 3459
type: B, layer: 1, pos: 2156
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2810
type: B, layer: 1, pos: 2098
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2127
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2602
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 691
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 3113
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2090
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 129
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 3114
type: B, layer: 1, pos: 2877
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2061
type: B, layer: 1, pos: 2200
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2175
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 302
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2424
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2646
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 2441
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 740
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2190
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2044
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2029
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 2030
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2205
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2145
type: B, layer: 1, pos: 2047
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2031
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 164
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2084
type: B, layer: 1, pos: 2099
type: B, layer: 1, pos: 2114
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2235
type: B, layer: 1, pos: 2236
type: B, layer: 1, pos: 2237
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2239
type: B, layer: 1, pos: 2834
type: B, layer: 1, pos: 2879

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 3472

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0145682, upper bound: 0.0146301
time: 7.38 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0146300
time: 10.16 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 23.55 seconds
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.55
Output dim: 0, lower bound: -0.0144320, upper bound: 0.0146298
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.55
Output dim: 0, lower bound: -0.0144936, upper bound: 0.0146300
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 23.55
Output dim: 0, lower bound: -0.0145682, upper bound: 0.0145657
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.55
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0145657
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 23.55
Output dim: 0, lower bound: -0.0145682, upper bound: 0.0145808
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.55
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0145807
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 23.55
Output dim: 0, lower bound: -0.0145682, upper bound: 0.0146151
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 23.55
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0146152
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 23.55
Output dim: 0, lower bound: -0.0145682, upper bound: 0.0146301
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 23.55
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0146300

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9214296, 1.2472771, 0.9212856, 1.2472457, -0.0307423, 0.0287448
1: -1.7818069, -0.8735242, -1.7819183, -0.8734070, -0.3132824, 0.3177759
2: -0.6261955, -0.1505359, -0.6263403, -0.1506550, -0.0478676, 0.0480093
3: -1.4627484, -0.9429598, -1.4627427, -0.9429173, -0.0869198, 0.0851607
4: -1.8161836, -0.9513952, -1.8163047, -0.9513491, -0.1604140, 0.1549423
5: -1.9003391, -1.3567022, -1.9003296, -1.3566587, -0.0882338, 0.0865795
6: -5.1290712, -4.2965837, -5.1288919, -4.2964997, -0.1408978, 0.1366640
7: -2.9590967, -2.1596298, -2.9590898, -2.1596045, -0.1344238, 0.1304235
8: -0.7256532, -0.3994496, -0.7255508, -0.3993502, -0.0298375, 0.0303796
9: -2.6078825, -1.6746111, -2.6078734, -1.6743982, -0.1805083, 0.1817110

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 3206
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 3102
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2027
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2810
type: A, layer: 1, pos: 2098
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2645
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2879

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2653

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0144216, upper bound: 0.0146229
time: 15.59 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0144237, upper bound: 0.0146230
time: 4.40 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9214316, 1.2474371, 0.9204397, 1.2474476, -0.0307742, 0.0297463
1: -1.7807095, -0.8735240, -1.7805781, -0.8734263, -0.3124290, 0.3166369
2: -0.6261933, -0.1500810, -0.6292563, -0.1500870, -0.0479552, 0.0516251
3: -1.4627483, -0.9432416, -1.4626684, -0.9432625, -0.0869004, 0.0854990
4: -1.8161787, -0.9516906, -1.8165013, -0.9517086, -0.1603996, 0.1556361
5: -1.9003382, -1.3576474, -1.9000890, -1.3578143, -0.0881864, 0.0879833
6: -5.1293840, -4.2965860, -5.1293006, -4.2927513, -0.1452388, 0.1367996
7: -2.9590967, -2.1611679, -2.9594018, -2.1614807, -0.1343516, 0.1337641
8: -0.7260472, -0.3994513, -0.7260470, -0.3972605, -0.0323157, 0.0304618
9: -2.6072648, -1.6746178, -2.6071198, -1.6740777, -0.1821418, 0.1816764

Time for backsubstitution: 5.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 3206
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 3102
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2027
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2810
type: A, layer: 1, pos: 2098
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2645
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2879

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2653

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0144832, upper bound: 0.0146230
time: 3.36 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0144853, upper bound: 0.0146226
time: 14.39 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.9215236, 1.2498417, 0.9206233, 1.2467765, -0.0287022, 0.0337710
1: -1.7832012, -0.8729311, -1.7798989, -0.8731178, -0.3177137, 0.3121788
2: -0.6268287, -0.1475884, -0.6298514, -0.1502754, -0.0476140, 0.0546639
3: -1.4622884, -0.9431088, -1.4612895, -0.9438586, -0.0858496, 0.0860201
4: -1.8146853, -0.9491738, -1.8139513, -0.9535179, -0.1558388, 0.1603571
5: -1.9001358, -1.3574404, -1.8990220, -1.3582318, -0.0873380, 0.0886924
6: -5.1288614, -4.2950616, -5.1264596, -4.2945557, -0.1426416, 0.1398269
7: -2.9579704, -2.1611464, -2.9574525, -2.1626134, -0.1310451, 0.1354245
8: -0.7280875, -0.3990994, -0.7259759, -0.3970363, -0.0346863, 0.0298132
9: -2.6115861, -1.6746206, -2.6067061, -1.6744754, -0.1873717, 0.1800007

Time for backsubstitution: 5.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 3206
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 3102
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2027
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2810
type: A, layer: 1, pos: 2098
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2645
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2879

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2653

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146182, upper bound: 0.0145587
time: 13.22 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146229, upper bound: 0.0145589
time: 4.70 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.9213808, 1.2501329, 0.9206233, 1.2470324, -0.0296361, 0.0333674
1: -1.7838802, -0.8731295, -1.7804840, -0.8731173, -0.3159201, 0.3166583
2: -0.6269240, -0.1474304, -0.6298518, -0.1501383, -0.0480208, 0.0544966
3: -1.4623429, -0.9430701, -1.4613370, -0.9438584, -0.0858176, 0.0861148
4: -1.8144470, -0.9491173, -1.8139536, -0.9534650, -0.1572372, 0.1599251
5: -1.9001281, -1.3574226, -1.8990219, -1.3582168, -0.0874745, 0.0886363
6: -5.1289349, -4.2950988, -5.1265244, -4.2945533, -0.1425755, 0.1402367
7: -2.9579117, -2.1610110, -2.9574528, -2.1624935, -0.1322376, 0.1348815
8: -0.7280962, -0.3990768, -0.7259766, -0.3970158, -0.0347166, 0.0298108
9: -2.6116567, -1.6746426, -2.6067684, -1.6744738, -0.1871464, 0.1805733

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 3206
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 3102
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2027
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2810
type: A, layer: 1, pos: 2098
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2645
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2879

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2653

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146182, upper bound: 0.0145738
time: 3.03 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146229, upper bound: 0.0145738
time: 2.92 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.9207701, 1.2498420, 0.9197805, 1.2471912, -0.0299580, 0.0334752
1: -1.7832088, -0.8729290, -1.7799935, -0.8731228, -0.3177245, 0.3122754
2: -0.6269053, -0.1475476, -0.6299121, -0.1502240, -0.0476794, 0.0550974
3: -1.4634466, -0.9430467, -1.4626204, -0.9430714, -0.0877906, 0.0854390
4: -1.8172815, -0.9491737, -1.8170100, -0.9517609, -0.1599989, 0.1591232
5: -1.9010630, -1.3574367, -1.9000897, -1.3576430, -0.0888587, 0.0880730
6: -5.1312838, -4.2949829, -5.1292348, -4.2924118, -0.1475300, 0.1382670
7: -2.9596636, -2.1611459, -2.9594018, -2.1614773, -0.1338364, 0.1343404
8: -0.7280905, -0.3988689, -0.7260477, -0.3967853, -0.0344047, 0.0305549
9: -2.6115866, -1.6737058, -2.6070585, -1.6733990, -0.1875637, 0.1812969

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 3206
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 3102
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2027
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2810
type: A, layer: 1, pos: 2098
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2645
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2879

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2653

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146181, upper bound: 0.0146081
time: 6.39 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146230, upper bound: 0.0146083
time: 2.90 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.9206256, 1.2499731, 0.9206264, 1.2472456, -0.0308597, 0.0320702
1: -1.7849841, -0.8731269, -1.7819200, -0.8731041, -0.3167853, 0.3178933
2: -0.6270033, -0.1478444, -0.6269968, -0.1506548, -0.0479982, 0.0513146
3: -1.4635017, -0.9427252, -1.4627432, -0.9427257, -0.0877781, 0.0851954
4: -1.8170481, -0.9488219, -1.8168163, -0.9513494, -0.1614150, 0.1579969
5: -1.9010557, -1.3564752, -1.9003303, -1.3564721, -0.0890425, 0.0866131
6: -5.1310453, -4.2950191, -5.1288919, -4.2961593, -0.1431233, 0.1385410
7: -2.9596062, -2.1594713, -2.9590905, -2.1594818, -0.1351011, 0.1304567
8: -0.7277051, -0.3988441, -0.7255515, -0.3988548, -0.0323482, 0.0304702
9: -2.6122758, -1.6737230, -2.6078732, -1.6737175, -0.1857049, 0.1819035

Time for backsubstitution: 5.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 3206
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 3102
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2027
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2810
type: A, layer: 1, pos: 2098
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2645
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2879

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2653

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145565, upper bound: 0.0146231
time: 5.97 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145613, upper bound: 0.0146229
time: 2.93 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.9206277, 1.2501332, 0.9197800, 1.2474475, -0.0308916, 0.0330718
1: -1.7838881, -0.8731277, -1.7805791, -0.8731228, -0.3159310, 0.3167542
2: -0.6270010, -0.1473898, -0.6299126, -0.1500872, -0.0480859, 0.0549304
3: -1.4635013, -0.9430069, -1.4626687, -0.9430709, -0.0877587, 0.0855337
4: -1.8170427, -0.9491172, -1.8170129, -0.9517080, -0.1614006, 0.1586909
5: -1.9010555, -1.3574200, -1.9000897, -1.3576281, -0.0889952, 0.0880168
6: -5.1313577, -4.2950206, -5.1292996, -4.2924099, -0.1474640, 0.1386768
7: -2.9596062, -2.1610098, -2.9594016, -2.1613579, -0.1350288, 0.1337972
8: -0.7280990, -0.3988456, -0.7260476, -0.3967653, -0.0344350, 0.0305524
9: -2.6116557, -1.6737287, -2.6071205, -1.6733973, -0.1873385, 0.1818693

Time for backsubstitution: 6.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 2305
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 3206
type: A, layer: 1, pos: 547
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2141
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2590
type: A, layer: 1, pos: 2026
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 534
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 3102
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2027
type: A, layer: 1, pos: 2155
type: A, layer: 1, pos: 3459
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 106
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2810
type: A, layer: 1, pos: 2098
type: A, layer: 1, pos: 782
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2127
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2602
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 2091
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2160
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 3113
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 129
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 3114
type: A, layer: 1, pos: 2877
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2061
type: A, layer: 1, pos: 2200
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 2175
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2424
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2646
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 766
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 2441
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 740
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2190
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 723
type: A, layer: 1, pos: 533
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 2030
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2205
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2145
type: A, layer: 1, pos: 2047
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2031
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2645
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 164
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2084
type: A, layer: 1, pos: 2099
type: A, layer: 1, pos: 2114
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2235
type: A, layer: 1, pos: 2236
type: A, layer: 1, pos: 2237
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2239
type: A, layer: 1, pos: 2834
type: A, layer: 1, pos: 2879

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2653

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146182, upper bound: 0.0146231
time: 4.35 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146230, upper bound: 0.0146228
time: 79.01 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 89.43 seconds
NS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 89.43
Output dim: 0, lower bound: -0.0144216, upper bound: 0.0146229
NS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 89.43
Output dim: 0, lower bound: -0.0144237, upper bound: 0.0146230
NS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 89.43
Output dim: 0, lower bound: -0.0144832, upper bound: 0.0146230
NS_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 89.43
Output dim: 0, lower bound: -0.0144853, upper bound: 0.0146226
NS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 89.43
Output dim: 0, lower bound: -0.0146182, upper bound: 0.0145587
NS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 89.43
Output dim: 0, lower bound: -0.0146229, upper bound: 0.0145589
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 89.43
Output dim: 0, lower bound: -0.0146182, upper bound: 0.0145738
NS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 89.43
Output dim: 0, lower bound: -0.0146229, upper bound: 0.0145738
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 89.43
Output dim: 0, lower bound: -0.0146181, upper bound: 0.0146081
NS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 89.43
Output dim: 0, lower bound: -0.0146230, upper bound: 0.0146083
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 89.43
Output dim: 0, lower bound: -0.0145565, upper bound: 0.0146231
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 89.43
Output dim: 0, lower bound: -0.0145613, upper bound: 0.0146229
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 89.43
Output dim: 0, lower bound: -0.0146182, upper bound: 0.0146231
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 89.43
Output dim: 0, lower bound: -0.0146230, upper bound: 0.0146228

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 32.11 + 1141.53 = 1173.64 seconds
