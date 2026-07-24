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
execution time: IAR + RelationalAnalysis = 7.74 + 24.88 = 32.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0146387, upper bound: 0.0146385

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3567
type: B, layer: 1, pos: 3567
type: A, layer: 1, pos: 335
type: B, layer: 1, pos: 335
type: A, layer: 1, pos: 2295
type: B, layer: 1, pos: 2295
type: A, layer: 1, pos: 3472
type: B, layer: 1, pos: 3472
type: A, layer: 1, pos: 2653
type: B, layer: 1, pos: 2653
type: A, layer: 1, pos: 2305
type: B, layer: 1, pos: 2305
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 3502
type: A, layer: 1, pos: 3206
type: B, layer: 1, pos: 3206
type: A, layer: 1, pos: 2058
type: B, layer: 1, pos: 2058
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 2041
type: B, layer: 1, pos: 2041
type: A, layer: 1, pos: 2141
type: B, layer: 1, pos: 2141
type: A, layer: 1, pos: 2590
type: B, layer: 1, pos: 2590
type: A, layer: 1, pos: 2040
type: B, layer: 1, pos: 2040
type: A, layer: 1, pos: 2026
type: B, layer: 1, pos: 2026
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 2042
type: B, layer: 1, pos: 2042
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 2222
type: B, layer: 1, pos: 2222
type: A, layer: 1, pos: 2539
type: B, layer: 1, pos: 2539
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 2025
type: B, layer: 1, pos: 2025
type: A, layer: 1, pos: 3102
type: B, layer: 1, pos: 3102
type: A, layer: 1, pos: 2112
type: B, layer: 1, pos: 2112
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2027
type: B, layer: 1, pos: 2027
type: A, layer: 1, pos: 3459
type: B, layer: 1, pos: 3459
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 2043
type: B, layer: 1, pos: 2043
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 2113
type: B, layer: 1, pos: 2113
type: A, layer: 1, pos: 2810
type: B, layer: 1, pos: 2810
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 2098
type: B, layer: 1, pos: 2098
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 2761
type: B, layer: 1, pos: 2761
type: A, layer: 1, pos: 2161
type: B, layer: 1, pos: 2161
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 2127
type: B, layer: 1, pos: 2127
type: A, layer: 1, pos: 2028
type: B, layer: 1, pos: 2028
type: A, layer: 1, pos: 2035
type: B, layer: 1, pos: 2035
type: A, layer: 1, pos: 2063
type: B, layer: 1, pos: 2063
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 2177
type: B, layer: 1, pos: 2177
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 2091
type: B, layer: 1, pos: 2091
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 2893
type: B, layer: 1, pos: 2893
type: A, layer: 1, pos: 815
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 2207
type: B, layer: 1, pos: 2207
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2160
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 3113
type: B, layer: 1, pos: 3113
type: A, layer: 1, pos: 2090
type: B, layer: 1, pos: 2090
type: A, layer: 1, pos: 2074
type: B, layer: 1, pos: 2074
type: A, layer: 1, pos: 2762
type: B, layer: 1, pos: 2762
type: A, layer: 1, pos: 2076
type: B, layer: 1, pos: 2076
type: A, layer: 1, pos: 2077
type: B, layer: 1, pos: 2077
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 2059
type: B, layer: 1, pos: 2059
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 3114
type: B, layer: 1, pos: 3114
type: A, layer: 1, pos: 2062
type: B, layer: 1, pos: 2062
type: A, layer: 1, pos: 2075
type: B, layer: 1, pos: 2075
type: A, layer: 1, pos: 2877
type: B, layer: 1, pos: 2877
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 2060
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 3292
type: B, layer: 1, pos: 3292
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 2176
type: B, layer: 1, pos: 2176
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 2200
type: B, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: B, layer: 1, pos: 2424
type: A, layer: 1, pos: 2049
type: B, layer: 1, pos: 2049
type: A, layer: 1, pos: 2034
type: B, layer: 1, pos: 2034
type: A, layer: 1, pos: 302
type: B, layer: 1, pos: 302
type: A, layer: 1, pos: 2048
type: B, layer: 1, pos: 2048
type: A, layer: 1, pos: 2033
type: B, layer: 1, pos: 2033
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 2650
type: B, layer: 1, pos: 2650
type: A, layer: 1, pos: 2441
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2646
type: B, layer: 1, pos: 2646
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 2795
type: B, layer: 1, pos: 2795
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2951
type: A, layer: 1, pos: 2201
type: B, layer: 1, pos: 2201
type: A, layer: 1, pos: 2221
type: B, layer: 1, pos: 2221
type: A, layer: 1, pos: 817
type: B, layer: 1, pos: 817
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 2191
type: B, layer: 1, pos: 2191
type: A, layer: 1, pos: 2202
type: B, layer: 1, pos: 2202
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 2190
type: B, layer: 1, pos: 2190
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 2192
type: B, layer: 1, pos: 2192
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 2421
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2044
type: B, layer: 1, pos: 2044
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 2217
type: B, layer: 1, pos: 2217
type: A, layer: 1, pos: 2220
type: B, layer: 1, pos: 2220
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 2029
type: B, layer: 1, pos: 2029
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 2045
type: B, layer: 1, pos: 2045
type: A, layer: 1, pos: 2616
type: B, layer: 1, pos: 2616
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 2030
type: B, layer: 1, pos: 2030
type: A, layer: 1, pos: 2206
type: B, layer: 1, pos: 2206
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 2205
type: B, layer: 1, pos: 2205
type: A, layer: 1, pos: 2218
type: B, layer: 1, pos: 2218
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 2047
type: B, layer: 1, pos: 2047
type: A, layer: 1, pos: 2031
type: B, layer: 1, pos: 2031
type: A, layer: 1, pos: 2046
type: B, layer: 1, pos: 2046
type: A, layer: 1, pos: 2645
type: B, layer: 1, pos: 2645
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
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

Time for candidate selection: 0.06 seconds

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
- Time for NS candidates: 7.01 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.01
Output dim: 0, lower bound: -0.0145022, upper bound: 0.0146385
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.01
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

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 335
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 2295
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 3472
type: A, layer: 1, pos: 3472
type: B, layer: 1, pos: 2653
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2305
type: B, layer: 1, pos: 2305
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 3502
type: A, layer: 1, pos: 3206
type: B, layer: 1, pos: 3206
type: A, layer: 1, pos: 2058
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 2041
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2141
type: B, layer: 1, pos: 2141
type: B, layer: 1, pos: 2590
type: A, layer: 1, pos: 2590
type: B, layer: 1, pos: 2040
type: A, layer: 1, pos: 2040
type: B, layer: 1, pos: 2026
type: A, layer: 1, pos: 2026
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 2042
type: A, layer: 1, pos: 2042
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 2222
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 2539
type: A, layer: 1, pos: 2539
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 2025
type: A, layer: 1, pos: 2025
type: B, layer: 1, pos: 3102
type: A, layer: 1, pos: 3102
type: B, layer: 1, pos: 2112
type: A, layer: 1, pos: 2112
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2027
type: A, layer: 1, pos: 2027
type: B, layer: 1, pos: 3459
type: A, layer: 1, pos: 3459
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 2156
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2043
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2113
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2810
type: B, layer: 1, pos: 2810
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 2098
type: A, layer: 1, pos: 2098
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: B, layer: 1, pos: 2761
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2161
type: B, layer: 1, pos: 2161
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2127
type: A, layer: 1, pos: 2127
type: B, layer: 1, pos: 2028
type: A, layer: 1, pos: 2028
type: B, layer: 1, pos: 2035
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2063
type: B, layer: 1, pos: 2063
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 2177
type: A, layer: 1, pos: 2177
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2091
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2893
type: B, layer: 1, pos: 2893
type: A, layer: 1, pos: 815
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2207
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 3113
type: B, layer: 1, pos: 3113
type: A, layer: 1, pos: 2090
type: B, layer: 1, pos: 2090
type: A, layer: 1, pos: 2074
type: B, layer: 1, pos: 2074
type: A, layer: 1, pos: 2762
type: B, layer: 1, pos: 2762
type: A, layer: 1, pos: 2076
type: B, layer: 1, pos: 2076
type: A, layer: 1, pos: 2077
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 2059
type: B, layer: 1, pos: 2059
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 3114
type: B, layer: 1, pos: 3114
type: A, layer: 1, pos: 2062
type: B, layer: 1, pos: 2062
type: A, layer: 1, pos: 2075
type: B, layer: 1, pos: 2075
type: A, layer: 1, pos: 2877
type: B, layer: 1, pos: 2877
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 2060
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 3292
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 2176
type: A, layer: 1, pos: 2176
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2200
type: A, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: A, layer: 1, pos: 2424
type: B, layer: 1, pos: 2049
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2034
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 302
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 2048
type: B, layer: 1, pos: 2048
type: A, layer: 1, pos: 2033
type: B, layer: 1, pos: 2033
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 2650
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2441
type: B, layer: 1, pos: 2646
type: A, layer: 1, pos: 2646
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2795
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2951
type: A, layer: 1, pos: 2201
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2221
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 817
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 2191
type: A, layer: 1, pos: 2191
type: B, layer: 1, pos: 2202
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 2190
type: A, layer: 1, pos: 2190
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 2192
type: A, layer: 1, pos: 2192
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2421
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2044
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2217
type: A, layer: 1, pos: 2217
type: B, layer: 1, pos: 2220
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2029
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 2045
type: A, layer: 1, pos: 2045
type: B, layer: 1, pos: 2616
type: A, layer: 1, pos: 2616
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 2030
type: A, layer: 1, pos: 2030
type: B, layer: 1, pos: 2206
type: A, layer: 1, pos: 2206
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 2205
type: A, layer: 1, pos: 2205
type: B, layer: 1, pos: 2218
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2047
type: B, layer: 1, pos: 2047
type: A, layer: 1, pos: 2031
type: B, layer: 1, pos: 2031
type: A, layer: 1, pos: 2046
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2645
type: A, layer: 1, pos: 2645
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
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

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 335

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145020, upper bound: 0.0145890
time: 406.24 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0145020, upper bound: 0.0146383
time: 8.99 seconds

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

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 335
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 2295
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 3472
type: A, layer: 1, pos: 3472
type: B, layer: 1, pos: 2653
type: A, layer: 1, pos: 2653
type: A, layer: 1, pos: 2305
type: B, layer: 1, pos: 2305
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 3502
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 3206
type: B, layer: 1, pos: 3206
type: B, layer: 1, pos: 2058
type: A, layer: 1, pos: 2058
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 2041
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2141
type: B, layer: 1, pos: 2141
type: A, layer: 1, pos: 2590
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2040
type: A, layer: 1, pos: 2040
type: B, layer: 1, pos: 2026
type: A, layer: 1, pos: 2026
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 2042
type: A, layer: 1, pos: 2042
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: A, layer: 1, pos: 2222
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2539
type: A, layer: 1, pos: 2539
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 2025
type: A, layer: 1, pos: 2025
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 3102
type: A, layer: 1, pos: 3102
type: B, layer: 1, pos: 2112
type: A, layer: 1, pos: 2112
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2027
type: A, layer: 1, pos: 2027
type: B, layer: 1, pos: 3459
type: A, layer: 1, pos: 3459
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 2559
type: B, layer: 1, pos: 2043
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2113
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2810
type: B, layer: 1, pos: 2810
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 2098
type: A, layer: 1, pos: 2098
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 2761
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2161
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2127
type: A, layer: 1, pos: 2127
type: B, layer: 1, pos: 2028
type: A, layer: 1, pos: 2028
type: B, layer: 1, pos: 2035
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2063
type: B, layer: 1, pos: 2063
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 2177
type: A, layer: 1, pos: 2177
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2091
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2893
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 815
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2207
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 3113
type: B, layer: 1, pos: 3113
type: A, layer: 1, pos: 2090
type: B, layer: 1, pos: 2090
type: A, layer: 1, pos: 2074
type: B, layer: 1, pos: 2074
type: B, layer: 1, pos: 2762
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 2076
type: B, layer: 1, pos: 2076
type: A, layer: 1, pos: 2077
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 2059
type: B, layer: 1, pos: 2059
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 3114
type: B, layer: 1, pos: 3114
type: A, layer: 1, pos: 2062
type: B, layer: 1, pos: 2062
type: A, layer: 1, pos: 2075
type: B, layer: 1, pos: 2075
type: A, layer: 1, pos: 2877
type: B, layer: 1, pos: 2877
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 2060
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 3292
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 2176
type: A, layer: 1, pos: 2176
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2200
type: A, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: A, layer: 1, pos: 2424
type: B, layer: 1, pos: 2049
type: A, layer: 1, pos: 2049
type: B, layer: 1, pos: 2034
type: A, layer: 1, pos: 2034
type: B, layer: 1, pos: 302
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 2048
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 2033
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 2650
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2441
type: B, layer: 1, pos: 2646
type: A, layer: 1, pos: 2646
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: B, layer: 1, pos: 2795
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2951
type: A, layer: 1, pos: 2201
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2221
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 817
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 2191
type: A, layer: 1, pos: 2191
type: B, layer: 1, pos: 2202
type: A, layer: 1, pos: 2202
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 2190
type: A, layer: 1, pos: 2190
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 2192
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2421
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2044
type: A, layer: 1, pos: 2044
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2217
type: A, layer: 1, pos: 2217
type: B, layer: 1, pos: 2220
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2029
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 2045
type: A, layer: 1, pos: 2045
type: B, layer: 1, pos: 2616
type: A, layer: 1, pos: 2616
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 2030
type: A, layer: 1, pos: 2030
type: B, layer: 1, pos: 2206
type: A, layer: 1, pos: 2206
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 2205
type: A, layer: 1, pos: 2205
type: B, layer: 1, pos: 2218
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2047
type: B, layer: 1, pos: 2047
type: A, layer: 1, pos: 2031
type: B, layer: 1, pos: 2031
type: A, layer: 1, pos: 2046
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2645
type: A, layer: 1, pos: 2645
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
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

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 335

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146383, upper bound: 0.0145890
time: 9.08 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146383, upper bound: 0.0146384
time: 5.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 20.47 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 20.47
Output dim: 0, lower bound: -0.0145020, upper bound: 0.0145890
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 20.47
Output dim: 0, lower bound: -0.0145020, upper bound: 0.0146383
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 20.47
Output dim: 0, lower bound: -0.0146383, upper bound: 0.0145890
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 20.47
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

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2295
type: B, layer: 1, pos: 2295
type: A, layer: 1, pos: 3472
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2653
type: A, layer: 1, pos: 2653
type: B, layer: 1, pos: 2305
type: A, layer: 1, pos: 2305
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 3206
type: A, layer: 1, pos: 3206
type: B, layer: 1, pos: 2058
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 2041
type: A, layer: 1, pos: 2041
type: B, layer: 1, pos: 2141
type: A, layer: 1, pos: 2141
type: B, layer: 1, pos: 2590
type: A, layer: 1, pos: 2590
type: B, layer: 1, pos: 2040
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2026
type: B, layer: 1, pos: 2026
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 2042
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 2222
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2539
type: B, layer: 1, pos: 3567
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2539
type: A, layer: 1, pos: 2025
type: B, layer: 1, pos: 2025
type: A, layer: 1, pos: 3102
type: B, layer: 1, pos: 3102
type: A, layer: 1, pos: 2112
type: B, layer: 1, pos: 2112
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2027
type: B, layer: 1, pos: 2027
type: A, layer: 1, pos: 3459
type: B, layer: 1, pos: 3459
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 2559
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2043
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2113
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2810
type: A, layer: 1, pos: 2810
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 2098
type: B, layer: 1, pos: 2098
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 2761
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 2161
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2127
type: B, layer: 1, pos: 2127
type: A, layer: 1, pos: 2028
type: B, layer: 1, pos: 2028
type: A, layer: 1, pos: 2035
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2063
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 2091
type: A, layer: 1, pos: 2177
type: B, layer: 1, pos: 2177
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2893
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2091
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2207
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2160
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2090
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 3113
type: A, layer: 1, pos: 3113
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 2074
type: A, layer: 1, pos: 2762
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2762
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2076
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 3114
type: A, layer: 1, pos: 3114
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2075
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2075
type: B, layer: 1, pos: 2877
type: A, layer: 1, pos: 2877
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 2060
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 3292
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 2176
type: B, layer: 1, pos: 2176
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2200
type: B, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: B, layer: 1, pos: 2424
type: A, layer: 1, pos: 2049
type: B, layer: 1, pos: 2049
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 2034
type: B, layer: 1, pos: 2034
type: A, layer: 1, pos: 2048
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 302
type: A, layer: 1, pos: 2033
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2795
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 2650
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2441
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2646
type: B, layer: 1, pos: 2646
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 2951
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 2201
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2221
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 817
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 2191
type: A, layer: 1, pos: 2202
type: B, layer: 1, pos: 2202
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 2190
type: B, layer: 1, pos: 2190
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 2192
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 2192
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2044
type: B, layer: 1, pos: 2044
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 2217
type: B, layer: 1, pos: 2217
type: A, layer: 1, pos: 2220
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2029
type: B, layer: 1, pos: 2029
type: A, layer: 1, pos: 2616
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 2045
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 2616
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 2030
type: B, layer: 1, pos: 2030
type: A, layer: 1, pos: 2206
type: B, layer: 1, pos: 2206
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 2205
type: B, layer: 1, pos: 2205
type: A, layer: 1, pos: 2218
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2047
type: A, layer: 1, pos: 2047
type: B, layer: 1, pos: 2031
type: A, layer: 1, pos: 2031
type: B, layer: 1, pos: 2046
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2645
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 808
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

Time for candidate selection: 0.06 seconds

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
time: 8.55 seconds

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

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2295
type: A, layer: 1, pos: 2295
type: B, layer: 1, pos: 3472
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 2653
type: B, layer: 1, pos: 2653
type: A, layer: 1, pos: 2305
type: B, layer: 1, pos: 2305
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 3502
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 3206
type: B, layer: 1, pos: 3206
type: A, layer: 1, pos: 2058
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 2041
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2141
type: B, layer: 1, pos: 2141
type: A, layer: 1, pos: 2590
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2040
type: A, layer: 1, pos: 2040
type: B, layer: 1, pos: 2026
type: A, layer: 1, pos: 2026
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 2042
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 2222
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2539
type: A, layer: 1, pos: 2539
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 2025
type: A, layer: 1, pos: 2025
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 3102
type: A, layer: 1, pos: 3102
type: B, layer: 1, pos: 2112
type: A, layer: 1, pos: 2112
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2027
type: A, layer: 1, pos: 2027
type: B, layer: 1, pos: 3459
type: A, layer: 1, pos: 3459
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2043
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2113
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2810
type: B, layer: 1, pos: 2810
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 2098
type: A, layer: 1, pos: 2098
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 2761
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2161
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2127
type: A, layer: 1, pos: 2127
type: B, layer: 1, pos: 2028
type: A, layer: 1, pos: 2028
type: B, layer: 1, pos: 2035
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2063
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 2177
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 2091
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 2893
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 815
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2207
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 3113
type: B, layer: 1, pos: 3113
type: A, layer: 1, pos: 2090
type: B, layer: 1, pos: 2090
type: A, layer: 1, pos: 2074
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 2074
type: A, layer: 1, pos: 2076
type: A, layer: 1, pos: 2762
type: B, layer: 1, pos: 2076
type: A, layer: 1, pos: 2077
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 2059
type: B, layer: 1, pos: 2059
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 3114
type: B, layer: 1, pos: 3114
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2075
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2075
type: A, layer: 1, pos: 2877
type: B, layer: 1, pos: 2877
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 2060
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 3292
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 2176
type: A, layer: 1, pos: 2176
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2200
type: A, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: A, layer: 1, pos: 2424
type: B, layer: 1, pos: 2049
type: A, layer: 1, pos: 2049
type: B, layer: 1, pos: 2034
type: A, layer: 1, pos: 2034
type: B, layer: 1, pos: 302
type: A, layer: 1, pos: 302
type: B, layer: 1, pos: 2048
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 2033
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 2650
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2441
type: B, layer: 1, pos: 2646
type: A, layer: 1, pos: 2646
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 2795
type: B, layer: 1, pos: 2951
type: A, layer: 1, pos: 2201
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2221
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 817
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 2191
type: B, layer: 1, pos: 2202
type: A, layer: 1, pos: 2202
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 2190
type: A, layer: 1, pos: 2190
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 2192
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2421
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2044
type: A, layer: 1, pos: 2044
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2217
type: A, layer: 1, pos: 2217
type: B, layer: 1, pos: 2220
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2029
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2045
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 2616
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 2030
type: A, layer: 1, pos: 2030
type: B, layer: 1, pos: 2206
type: A, layer: 1, pos: 2206
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 2205
type: A, layer: 1, pos: 2205
type: B, layer: 1, pos: 2218
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2047
type: B, layer: 1, pos: 2047
type: A, layer: 1, pos: 2031
type: B, layer: 1, pos: 2031
type: A, layer: 1, pos: 2046
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2645
type: A, layer: 1, pos: 2645
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
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

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2295

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146151, upper bound: 0.0145807
time: 6.81 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0145809
time: 4.70 seconds

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

Time for backsubstitution: 6.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2295
type: B, layer: 1, pos: 2295
type: A, layer: 1, pos: 3472
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2653
type: A, layer: 1, pos: 2653
type: B, layer: 1, pos: 2305
type: A, layer: 1, pos: 2305
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 3206
type: A, layer: 1, pos: 3206
type: B, layer: 1, pos: 2058
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 2041
type: A, layer: 1, pos: 2041
type: B, layer: 1, pos: 2141
type: A, layer: 1, pos: 2141
type: B, layer: 1, pos: 2590
type: A, layer: 1, pos: 2590
type: B, layer: 1, pos: 2040
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2026
type: B, layer: 1, pos: 2026
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2042
type: A, layer: 1, pos: 2042
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 2222
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 2025
type: A, layer: 1, pos: 2025
type: B, layer: 1, pos: 3567
type: A, layer: 1, pos: 3102
type: B, layer: 1, pos: 3102
type: B, layer: 1, pos: 2112
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2027
type: B, layer: 1, pos: 2027
type: A, layer: 1, pos: 3459
type: B, layer: 1, pos: 3459
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 2559
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 2043
type: A, layer: 1, pos: 2043
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 2113
type: A, layer: 1, pos: 2113
type: B, layer: 1, pos: 2810
type: A, layer: 1, pos: 2810
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 2098
type: B, layer: 1, pos: 2098
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 2761
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2161
type: B, layer: 1, pos: 2161
type: A, layer: 1, pos: 335
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2127
type: B, layer: 1, pos: 2127
type: A, layer: 1, pos: 2028
type: B, layer: 1, pos: 2028
type: A, layer: 1, pos: 2035
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2063
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 2091
type: A, layer: 1, pos: 2177
type: B, layer: 1, pos: 2177
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2893
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2091
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2207
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2160
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2090
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 3113
type: A, layer: 1, pos: 3113
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 2074
type: A, layer: 1, pos: 2762
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2762
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2076
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 3114
type: A, layer: 1, pos: 3114
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2075
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2075
type: B, layer: 1, pos: 2877
type: A, layer: 1, pos: 2877
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 2060
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 3292
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 2176
type: B, layer: 1, pos: 2176
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2200
type: B, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: B, layer: 1, pos: 2424
type: A, layer: 1, pos: 2049
type: B, layer: 1, pos: 2049
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 2034
type: B, layer: 1, pos: 2034
type: A, layer: 1, pos: 2048
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 302
type: A, layer: 1, pos: 2033
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2795
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 2650
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2441
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2646
type: B, layer: 1, pos: 2646
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 2951
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 2201
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2221
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 817
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 2191
type: A, layer: 1, pos: 2202
type: B, layer: 1, pos: 2202
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 2190
type: B, layer: 1, pos: 2190
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 2192
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 2192
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2044
type: B, layer: 1, pos: 2044
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 2217
type: B, layer: 1, pos: 2217
type: A, layer: 1, pos: 2220
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2029
type: B, layer: 1, pos: 2029
type: A, layer: 1, pos: 2616
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 2045
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 2616
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 2030
type: B, layer: 1, pos: 2030
type: A, layer: 1, pos: 2206
type: B, layer: 1, pos: 2206
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 2205
type: B, layer: 1, pos: 2205
type: A, layer: 1, pos: 2218
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2047
type: A, layer: 1, pos: 2047
type: B, layer: 1, pos: 2031
type: A, layer: 1, pos: 2031
type: B, layer: 1, pos: 2046
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2645
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 808
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2295

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146300, upper bound: 0.0146151
time: 20.01 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0146299
time: 7.94 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 34.06 seconds
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 34.06
Output dim: 0, lower bound: -0.0144936, upper bound: 0.0146153
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 34.06
Output dim: 0, lower bound: -0.0144936, upper bound: 0.0146298
NS_A2_B1_B1, status: Status.VERIFIED, split count: 3, time: 34.06
Output dim: 0, lower bound: -0.0146151, upper bound: 0.0145807
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 34.06
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0145809
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 34.06
Output dim: 0, lower bound: -0.0146300, upper bound: 0.0146151
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 34.06
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

Time for backsubstitution: 5.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3472
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2653
type: A, layer: 1, pos: 2653
type: B, layer: 1, pos: 2305
type: A, layer: 1, pos: 2305
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 3206
type: A, layer: 1, pos: 3206
type: B, layer: 1, pos: 2058
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 2041
type: A, layer: 1, pos: 2041
type: B, layer: 1, pos: 2141
type: A, layer: 1, pos: 2141
type: B, layer: 1, pos: 2590
type: A, layer: 1, pos: 2590
type: B, layer: 1, pos: 2040
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2026
type: B, layer: 1, pos: 2026
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 2042
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 2222
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2539
type: B, layer: 1, pos: 3567
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2539
type: A, layer: 1, pos: 2025
type: B, layer: 1, pos: 2025
type: A, layer: 1, pos: 3102
type: B, layer: 1, pos: 3102
type: A, layer: 1, pos: 2112
type: B, layer: 1, pos: 2112
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2027
type: B, layer: 1, pos: 2027
type: A, layer: 1, pos: 3459
type: B, layer: 1, pos: 3459
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 2559
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2043
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2113
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2810
type: A, layer: 1, pos: 2810
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 2098
type: B, layer: 1, pos: 2098
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 2761
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 2161
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2127
type: B, layer: 1, pos: 2127
type: A, layer: 1, pos: 2028
type: B, layer: 1, pos: 2028
type: A, layer: 1, pos: 2035
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2063
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 2091
type: A, layer: 1, pos: 2177
type: B, layer: 1, pos: 2177
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2893
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2091
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2207
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2160
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2090
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 3113
type: A, layer: 1, pos: 3113
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 2074
type: A, layer: 1, pos: 2762
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2762
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2076
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 3114
type: A, layer: 1, pos: 3114
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2075
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2075
type: B, layer: 1, pos: 2877
type: A, layer: 1, pos: 2877
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 2060
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 3292
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 2176
type: B, layer: 1, pos: 2176
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2200
type: B, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: B, layer: 1, pos: 2424
type: A, layer: 1, pos: 2049
type: B, layer: 1, pos: 2049
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 2034
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 2048
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 302
type: A, layer: 1, pos: 2033
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2795
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 2650
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2441
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2646
type: B, layer: 1, pos: 2646
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 2951
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2795
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2221
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 817
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 2191
type: A, layer: 1, pos: 2202
type: B, layer: 1, pos: 2202
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 2190
type: B, layer: 1, pos: 2190
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 2192
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 2192
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2044
type: B, layer: 1, pos: 2044
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 2217
type: B, layer: 1, pos: 2217
type: A, layer: 1, pos: 2220
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2029
type: B, layer: 1, pos: 2029
type: A, layer: 1, pos: 2616
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 2045
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 2616
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 2030
type: B, layer: 1, pos: 2030
type: A, layer: 1, pos: 2206
type: B, layer: 1, pos: 2206
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 2205
type: B, layer: 1, pos: 2205
type: A, layer: 1, pos: 2218
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2047
type: A, layer: 1, pos: 2047
type: B, layer: 1, pos: 2031
type: A, layer: 1, pos: 2031
type: B, layer: 1, pos: 2046
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2645
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 808
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3472

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0144936, upper bound: 0.0145681
time: 10.15 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0144936, upper bound: 0.0146298
time: 447.51 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.9213789, 1.2501438, 0.9214695, 1.2470229, -0.0288157, 0.0335154
1: -1.7851350, -0.8731294, -1.7817509, -0.8730984, -0.3214163, 0.3132510
2: -0.6269261, -0.1474038, -0.6269360, -0.1501153, -0.0482350, 0.0518849
3: -1.4623437, -0.9427856, -1.4614116, -0.9435106, -0.0859641, 0.0857058
4: -1.8144531, -0.9487159, -1.8137575, -0.9531032, -0.1558614, 0.1609146
5: -1.9001286, -1.3564668, -1.8992629, -1.3570565, -0.0876624, 0.0876051
6: -5.1293693, -4.2950964, -5.1270151, -4.2983027, -0.1394491, 0.1405742
7: -2.9579117, -2.1592915, -2.9571412, -2.1604662, -0.1317005, 0.1333562
8: -0.7280982, -0.3990744, -0.7259787, -0.3991061, -0.0329963, 0.0302601
9: -2.6123953, -1.6746368, -2.6076317, -1.6747937, -0.1863813, 0.1803209

Time for backsubstitution: 5.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3472
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 2653
type: B, layer: 1, pos: 2653
type: A, layer: 1, pos: 2305
type: B, layer: 1, pos: 2305
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 3502
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 3206
type: B, layer: 1, pos: 3206
type: A, layer: 1, pos: 2058
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 2041
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2141
type: B, layer: 1, pos: 2141
type: A, layer: 1, pos: 2590
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2040
type: A, layer: 1, pos: 2040
type: B, layer: 1, pos: 2026
type: A, layer: 1, pos: 2026
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 2042
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 2222
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2539
type: A, layer: 1, pos: 2539
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 2025
type: A, layer: 1, pos: 2025
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 3102
type: A, layer: 1, pos: 3102
type: B, layer: 1, pos: 2112
type: A, layer: 1, pos: 2112
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2027
type: A, layer: 1, pos: 2027
type: B, layer: 1, pos: 3459
type: A, layer: 1, pos: 3459
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2043
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2113
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2810
type: B, layer: 1, pos: 2810
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 2098
type: A, layer: 1, pos: 2098
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 2761
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2161
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2127
type: A, layer: 1, pos: 2127
type: B, layer: 1, pos: 2028
type: A, layer: 1, pos: 2028
type: B, layer: 1, pos: 2035
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2063
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: A, layer: 1, pos: 335
type: B, layer: 1, pos: 2177
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 2091
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 2893
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 815
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2207
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 3113
type: B, layer: 1, pos: 3113
type: A, layer: 1, pos: 2090
type: B, layer: 1, pos: 2090
type: A, layer: 1, pos: 2074
type: B, layer: 1, pos: 2762
type: A, layer: 1, pos: 2076
type: B, layer: 1, pos: 2074
type: A, layer: 1, pos: 2762
type: B, layer: 1, pos: 2076
type: A, layer: 1, pos: 2077
type: B, layer: 1, pos: 2077
type: A, layer: 1, pos: 2295
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 2059
type: B, layer: 1, pos: 2059
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 3114
type: B, layer: 1, pos: 3114
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2075
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2075
type: A, layer: 1, pos: 2877
type: B, layer: 1, pos: 2877
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 2060
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 3292
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 2176
type: A, layer: 1, pos: 2176
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2200
type: A, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: A, layer: 1, pos: 2424
type: B, layer: 1, pos: 2049
type: A, layer: 1, pos: 2049
type: B, layer: 1, pos: 2034
type: A, layer: 1, pos: 2034
type: B, layer: 1, pos: 302
type: A, layer: 1, pos: 302
type: B, layer: 1, pos: 2048
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 2033
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 2650
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2441
type: B, layer: 1, pos: 2646
type: A, layer: 1, pos: 2646
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2951
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 2201
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2221
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 817
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 2191
type: B, layer: 1, pos: 2202
type: A, layer: 1, pos: 2202
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 2190
type: A, layer: 1, pos: 2190
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 2192
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2421
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2044
type: A, layer: 1, pos: 2044
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2217
type: A, layer: 1, pos: 2217
type: B, layer: 1, pos: 2220
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2029
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2045
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 2616
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 2030
type: A, layer: 1, pos: 2030
type: B, layer: 1, pos: 2206
type: A, layer: 1, pos: 2206
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 2205
type: A, layer: 1, pos: 2205
type: B, layer: 1, pos: 2218
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2047
type: B, layer: 1, pos: 2047
type: A, layer: 1, pos: 2031
type: B, layer: 1, pos: 2031
type: A, layer: 1, pos: 2046
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2645
type: A, layer: 1, pos: 2645
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
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

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3472

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145682, upper bound: 0.0145807
time: 5.34 seconds

## Relational analysis of NS_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0145809
time: 23.37 seconds

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

Time for backsubstitution: 5.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3472
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2653
type: A, layer: 1, pos: 2653
type: B, layer: 1, pos: 2305
type: A, layer: 1, pos: 2305
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 3206
type: A, layer: 1, pos: 3206
type: B, layer: 1, pos: 2058
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 2041
type: A, layer: 1, pos: 2041
type: B, layer: 1, pos: 2141
type: A, layer: 1, pos: 2141
type: B, layer: 1, pos: 2590
type: A, layer: 1, pos: 2590
type: B, layer: 1, pos: 2040
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2026
type: B, layer: 1, pos: 2026
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2042
type: A, layer: 1, pos: 2042
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 2222
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 2025
type: A, layer: 1, pos: 2025
type: B, layer: 1, pos: 3567
type: A, layer: 1, pos: 3102
type: B, layer: 1, pos: 3102
type: B, layer: 1, pos: 2112
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2027
type: B, layer: 1, pos: 2027
type: A, layer: 1, pos: 3459
type: B, layer: 1, pos: 3459
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 2559
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 2043
type: A, layer: 1, pos: 2043
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 2113
type: A, layer: 1, pos: 2113
type: B, layer: 1, pos: 2810
type: A, layer: 1, pos: 2810
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 2098
type: B, layer: 1, pos: 2098
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 2761
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2161
type: B, layer: 1, pos: 2161
type: A, layer: 1, pos: 335
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2127
type: B, layer: 1, pos: 2127
type: A, layer: 1, pos: 2028
type: B, layer: 1, pos: 2028
type: A, layer: 1, pos: 2035
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2063
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 2091
type: A, layer: 1, pos: 2177
type: B, layer: 1, pos: 2177
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2893
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2091
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2207
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2160
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2090
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 3113
type: A, layer: 1, pos: 3113
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 2074
type: A, layer: 1, pos: 2762
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2762
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 2090
type: B, layer: 1, pos: 2295
type: A, layer: 1, pos: 2076
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 2059
type: A, layer: 1, pos: 2059
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 3114
type: A, layer: 1, pos: 3114
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2075
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2075
type: B, layer: 1, pos: 2877
type: A, layer: 1, pos: 2877
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 2060
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 3292
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 2176
type: B, layer: 1, pos: 2176
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2200
type: B, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: B, layer: 1, pos: 2424
type: A, layer: 1, pos: 2049
type: B, layer: 1, pos: 2049
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 2034
type: B, layer: 1, pos: 2034
type: A, layer: 1, pos: 2048
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 302
type: A, layer: 1, pos: 2033
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2795
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 2650
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2441
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2646
type: B, layer: 1, pos: 2646
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 2951
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 2201
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2221
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 817
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 2191
type: A, layer: 1, pos: 2202
type: B, layer: 1, pos: 2202
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 2190
type: B, layer: 1, pos: 2190
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 2192
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 2192
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2044
type: B, layer: 1, pos: 2044
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 2217
type: B, layer: 1, pos: 2217
type: A, layer: 1, pos: 2220
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2029
type: B, layer: 1, pos: 2029
type: A, layer: 1, pos: 2616
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 2045
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 2616
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 2030
type: B, layer: 1, pos: 2030
type: A, layer: 1, pos: 2206
type: B, layer: 1, pos: 2206
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 2205
type: B, layer: 1, pos: 2205
type: A, layer: 1, pos: 2218
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2047
type: A, layer: 1, pos: 2047
type: B, layer: 1, pos: 2031
type: A, layer: 1, pos: 2031
type: B, layer: 1, pos: 2046
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2645
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 808
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3472

## Relational analysis of NS_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0145537
time: 2.87 seconds

## Relational analysis of NS_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0146152
time: 9.67 seconds

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

Time for backsubstitution: 5.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3472
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2653
type: A, layer: 1, pos: 2653
type: B, layer: 1, pos: 2305
type: A, layer: 1, pos: 2305
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 3206
type: A, layer: 1, pos: 3206
type: B, layer: 1, pos: 2058
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 2041
type: A, layer: 1, pos: 2041
type: B, layer: 1, pos: 2141
type: A, layer: 1, pos: 2141
type: B, layer: 1, pos: 2590
type: A, layer: 1, pos: 2590
type: B, layer: 1, pos: 2040
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2026
type: B, layer: 1, pos: 2026
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2042
type: A, layer: 1, pos: 2042
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 2222
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 2025
type: A, layer: 1, pos: 2025
type: B, layer: 1, pos: 3567
type: A, layer: 1, pos: 3102
type: B, layer: 1, pos: 3102
type: B, layer: 1, pos: 2112
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2027
type: B, layer: 1, pos: 2027
type: A, layer: 1, pos: 3459
type: B, layer: 1, pos: 3459
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 2559
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 2043
type: A, layer: 1, pos: 2043
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 2113
type: A, layer: 1, pos: 2113
type: B, layer: 1, pos: 2810
type: A, layer: 1, pos: 2810
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 2098
type: B, layer: 1, pos: 2098
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 2761
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 2161
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2127
type: B, layer: 1, pos: 2127
type: A, layer: 1, pos: 2028
type: B, layer: 1, pos: 2028
type: A, layer: 1, pos: 2035
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2063
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 2091
type: A, layer: 1, pos: 2177
type: B, layer: 1, pos: 2177
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2893
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2091
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2207
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2160
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2090
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 3113
type: A, layer: 1, pos: 3113
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 2074
type: A, layer: 1, pos: 2762
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2762
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2076
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 3114
type: A, layer: 1, pos: 3114
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2075
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2075
type: B, layer: 1, pos: 2877
type: A, layer: 1, pos: 2877
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 2060
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 3292
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 2176
type: B, layer: 1, pos: 2176
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2200
type: B, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: B, layer: 1, pos: 2424
type: A, layer: 1, pos: 2049
type: B, layer: 1, pos: 2049
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 2034
type: B, layer: 1, pos: 2034
type: A, layer: 1, pos: 2048
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 302
type: A, layer: 1, pos: 2033
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2795
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 2650
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2441
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2646
type: B, layer: 1, pos: 2646
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 2951
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 2201
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2221
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 817
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 2191
type: A, layer: 1, pos: 2202
type: B, layer: 1, pos: 2202
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 2190
type: B, layer: 1, pos: 2190
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 2192
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 2192
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2044
type: B, layer: 1, pos: 2044
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 2217
type: B, layer: 1, pos: 2217
type: A, layer: 1, pos: 2220
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2029
type: B, layer: 1, pos: 2029
type: A, layer: 1, pos: 2616
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 2045
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 2616
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 2030
type: B, layer: 1, pos: 2030
type: A, layer: 1, pos: 2206
type: B, layer: 1, pos: 2206
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 2205
type: B, layer: 1, pos: 2205
type: A, layer: 1, pos: 2218
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2047
type: A, layer: 1, pos: 2047
type: B, layer: 1, pos: 2031
type: A, layer: 1, pos: 2031
type: B, layer: 1, pos: 2046
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2645
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 808
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3472

## Relational analysis of NS_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0145682
time: 65.43 seconds

## Relational analysis of NS_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0145680
time: 35.56 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 106.95 seconds
NS_A1_B2_A2_A1, status: Status.VERIFIED, split count: 4, time: 106.95
Output dim: 0, lower bound: -0.0144936, upper bound: 0.0145681
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 106.95
Output dim: 0, lower bound: -0.0144936, upper bound: 0.0146298
NS_A2_B1_B2_B1, status: Status.VERIFIED, split count: 4, time: 106.95
Output dim: 0, lower bound: -0.0145682, upper bound: 0.0145807
NS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 106.95
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0145809
NS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 106.95
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0145537
NS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 106.95
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0146152
NS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 106.95
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0145682
NS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 106.95
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0145680

## BFS NS instance: NS_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: 0.9205834, 1.2474372, 0.9212876, 1.2474476, -0.0317839, 0.0287366
1: -1.7804554, -0.8735427, -1.7808321, -0.8734075, -0.3121331, 0.3169326
2: -0.6291113, -0.1500858, -0.6263379, -0.1500824, -0.0516331, 0.0479472
3: -1.4626741, -0.9433057, -1.4627428, -0.9431990, -0.0872640, 0.0851354
4: -1.8163792, -0.9517668, -1.8162998, -0.9516314, -0.1611364, 0.1548993
5: -1.9000980, -1.3578601, -1.9003292, -1.3576016, -0.0896963, 0.0864735
6: -5.1292906, -4.2928352, -5.1293926, -4.2965021, -0.1408406, 0.1411979
7: -2.9594083, -2.1615441, -2.9590898, -2.1611042, -0.1379019, 0.1302136
8: -0.7260467, -0.3973603, -0.7260469, -0.3993521, -0.0298173, 0.0329603
9: -2.6071024, -1.6742914, -2.6072817, -1.6744039, -0.1804233, 0.1833951

Time for backsubstitution: 5.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2653
type: A, layer: 1, pos: 2653
type: B, layer: 1, pos: 2305
type: A, layer: 1, pos: 2305
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 3206
type: A, layer: 1, pos: 3206
type: B, layer: 1, pos: 2058
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 2041
type: A, layer: 1, pos: 2041
type: B, layer: 1, pos: 2141
type: A, layer: 1, pos: 2141
type: B, layer: 1, pos: 2590
type: A, layer: 1, pos: 2590
type: B, layer: 1, pos: 2040
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2026
type: B, layer: 1, pos: 2026
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2042
type: A, layer: 1, pos: 2042
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 2222
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2539
type: B, layer: 1, pos: 3567
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2539
type: A, layer: 1, pos: 2025
type: B, layer: 1, pos: 2025
type: A, layer: 1, pos: 3102
type: B, layer: 1, pos: 3102
type: A, layer: 1, pos: 2112
type: B, layer: 1, pos: 2112
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2027
type: B, layer: 1, pos: 2027
type: A, layer: 1, pos: 3459
type: B, layer: 1, pos: 3459
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 2559
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: A, layer: 1, pos: 2043
type: B, layer: 1, pos: 2043
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2113
type: B, layer: 1, pos: 2113
type: B, layer: 1, pos: 2810
type: A, layer: 1, pos: 2810
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 2098
type: B, layer: 1, pos: 2098
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 2761
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 2161
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2127
type: B, layer: 1, pos: 2127
type: A, layer: 1, pos: 2028
type: B, layer: 1, pos: 2028
type: A, layer: 1, pos: 2035
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2063
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 2091
type: A, layer: 1, pos: 2177
type: B, layer: 1, pos: 2177
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2893
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2091
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2207
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2160
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2090
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 3113
type: A, layer: 1, pos: 3113
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 2074
type: A, layer: 1, pos: 2762
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2762
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2076
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 3114
type: A, layer: 1, pos: 3114
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2075
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2075
type: B, layer: 1, pos: 2877
type: A, layer: 1, pos: 2877
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 2060
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 3292
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 2176
type: B, layer: 1, pos: 2176
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2200
type: B, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: B, layer: 1, pos: 2424
type: A, layer: 1, pos: 2049
type: B, layer: 1, pos: 2049
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 2034
type: B, layer: 1, pos: 2034
type: A, layer: 1, pos: 2048
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 302
type: A, layer: 1, pos: 2033
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2795
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 2650
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2441
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2646
type: B, layer: 1, pos: 2646
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 2951
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 2201
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2221
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 817
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 2191
type: A, layer: 1, pos: 2202
type: B, layer: 1, pos: 2202
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 2190
type: B, layer: 1, pos: 2190
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 2192
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 2192
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2044
type: B, layer: 1, pos: 2044
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 2217
type: B, layer: 1, pos: 2217
type: A, layer: 1, pos: 2220
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2029
type: B, layer: 1, pos: 2029
type: A, layer: 1, pos: 2616
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 2045
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 2616
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 2030
type: B, layer: 1, pos: 2030
type: A, layer: 1, pos: 2206
type: B, layer: 1, pos: 2206
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 2205
type: B, layer: 1, pos: 2205
type: A, layer: 1, pos: 2218
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2047
type: A, layer: 1, pos: 2047
type: B, layer: 1, pos: 2031
type: A, layer: 1, pos: 2031
type: B, layer: 1, pos: 2046
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2645
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 808
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

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2653

## Relational analysis of NS_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0144853, upper bound: 0.0146182
time: 2.89 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0144853, upper bound: 0.0146230
time: 4.17 seconds

## BFS NS instance: NS_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: 0.9213808, 1.2501432, 0.9206232, 1.2470223, -0.0286469, 0.0343564
1: -1.7840018, -0.8731295, -1.7803624, -0.8731178, -0.3205197, 0.3120583
2: -0.6269240, -0.1474267, -0.6298518, -0.1501423, -0.0475947, 0.0549227
3: -1.4623437, -0.9430701, -1.4613369, -0.9438584, -0.0859130, 0.0860195
4: -1.8144484, -0.9490584, -1.8139533, -0.9535244, -0.1556923, 0.1614826
5: -1.9001281, -1.3574184, -1.8990219, -1.3582211, -0.0873293, 0.0887818
6: -5.1289535, -4.2950988, -5.1265063, -4.2945538, -0.1430434, 0.1397702
7: -2.9579117, -2.1609781, -2.9574528, -2.1625268, -0.1309565, 0.1361625
8: -0.7280962, -0.3990760, -0.7259765, -0.3970163, -0.0346874, 0.0298439
9: -2.6116734, -1.6746421, -2.6067510, -1.6744738, -0.1877493, 0.1799704

Time for backsubstitution: 5.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2653
type: B, layer: 1, pos: 2653
type: A, layer: 1, pos: 2305
type: B, layer: 1, pos: 2305
type: A, layer: 1, pos: 2347
type: B, layer: 1, pos: 2347
type: B, layer: 1, pos: 3502
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 3206
type: B, layer: 1, pos: 3206
type: A, layer: 1, pos: 2058
type: B, layer: 1, pos: 2058
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 547
type: B, layer: 1, pos: 2041
type: A, layer: 1, pos: 2041
type: A, layer: 1, pos: 2141
type: B, layer: 1, pos: 2141
type: A, layer: 1, pos: 2590
type: B, layer: 1, pos: 2590
type: B, layer: 1, pos: 2040
type: A, layer: 1, pos: 2040
type: B, layer: 1, pos: 2026
type: A, layer: 1, pos: 2026
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 2042
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 2222
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 2539
type: A, layer: 1, pos: 2539
type: B, layer: 1, pos: 534
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 2025
type: A, layer: 1, pos: 2025
type: B, layer: 1, pos: 3567
type: B, layer: 1, pos: 3102
type: A, layer: 1, pos: 3102
type: B, layer: 1, pos: 2112
type: A, layer: 1, pos: 2112
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2027
type: A, layer: 1, pos: 2027
type: B, layer: 1, pos: 3459
type: A, layer: 1, pos: 3459
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 750
type: B, layer: 1, pos: 2043
type: A, layer: 1, pos: 2043
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2113
type: A, layer: 1, pos: 2113
type: A, layer: 1, pos: 2810
type: B, layer: 1, pos: 2810
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 2098
type: A, layer: 1, pos: 2098
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 2761
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2161
type: A, layer: 1, pos: 2161
type: A, layer: 1, pos: 801
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 2127
type: A, layer: 1, pos: 2127
type: B, layer: 1, pos: 2028
type: A, layer: 1, pos: 2028
type: B, layer: 1, pos: 2035
type: A, layer: 1, pos: 2035
type: A, layer: 1, pos: 2063
type: B, layer: 1, pos: 2063
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 2177
type: A, layer: 1, pos: 2177
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 2091
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 692
type: A, layer: 1, pos: 692
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 2091
type: B, layer: 1, pos: 2893
type: A, layer: 1, pos: 2893
type: A, layer: 1, pos: 815
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 736
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 158
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 741
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2207
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2160
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 3113
type: B, layer: 1, pos: 3113
type: A, layer: 1, pos: 2090
type: B, layer: 1, pos: 2090
type: A, layer: 1, pos: 2074
type: B, layer: 1, pos: 2762
type: A, layer: 1, pos: 2076
type: B, layer: 1, pos: 2074
type: A, layer: 1, pos: 2762
type: B, layer: 1, pos: 2076
type: A, layer: 1, pos: 2077
type: B, layer: 1, pos: 2077
type: A, layer: 1, pos: 2295
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 691
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 3472
type: A, layer: 1, pos: 2059
type: B, layer: 1, pos: 2059
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 3114
type: B, layer: 1, pos: 3114
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2075
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2075
type: A, layer: 1, pos: 2877
type: B, layer: 1, pos: 2877
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 2060
type: B, layer: 1, pos: 2060
type: B, layer: 1, pos: 3292
type: A, layer: 1, pos: 3292
type: A, layer: 1, pos: 816
type: B, layer: 1, pos: 2176
type: A, layer: 1, pos: 2176
type: B, layer: 1, pos: 816
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2200
type: A, layer: 1, pos: 2200
type: B, layer: 1, pos: 2424
type: A, layer: 1, pos: 2424
type: B, layer: 1, pos: 2049
type: A, layer: 1, pos: 2049
type: B, layer: 1, pos: 2034
type: A, layer: 1, pos: 2034
type: B, layer: 1, pos: 302
type: A, layer: 1, pos: 302
type: B, layer: 1, pos: 2048
type: A, layer: 1, pos: 2048
type: A, layer: 1, pos: 139
type: B, layer: 1, pos: 2033
type: A, layer: 1, pos: 2033
type: A, layer: 1, pos: 803
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 139
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 2650
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2441
type: B, layer: 1, pos: 2646
type: A, layer: 1, pos: 2646
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2184
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2951
type: A, layer: 1, pos: 2795
type: A, layer: 1, pos: 2201
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 2221
type: A, layer: 1, pos: 2221
type: A, layer: 1, pos: 817
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 861
type: A, layer: 1, pos: 861
type: A, layer: 1, pos: 2191
type: B, layer: 1, pos: 2202
type: A, layer: 1, pos: 2202
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 2190
type: A, layer: 1, pos: 2190
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 2192
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 723
type: B, layer: 1, pos: 868
type: A, layer: 1, pos: 868
type: A, layer: 1, pos: 2421
type: B, layer: 1, pos: 2421
type: B, layer: 1, pos: 2044
type: A, layer: 1, pos: 2044
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2217
type: A, layer: 1, pos: 2217
type: B, layer: 1, pos: 2220
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 156
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2029
type: A, layer: 1, pos: 2029
type: A, layer: 1, pos: 511
type: B, layer: 1, pos: 511
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 2045
type: A, layer: 1, pos: 2045
type: A, layer: 1, pos: 2616
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 2030
type: A, layer: 1, pos: 2030
type: B, layer: 1, pos: 2206
type: A, layer: 1, pos: 2206
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 2205
type: A, layer: 1, pos: 2205
type: B, layer: 1, pos: 2218
type: A, layer: 1, pos: 2218
type: A, layer: 1, pos: 2032
type: B, layer: 1, pos: 2032
type: B, layer: 1, pos: 877
type: A, layer: 1, pos: 877
type: A, layer: 1, pos: 2047
type: B, layer: 1, pos: 2047
type: A, layer: 1, pos: 2031
type: B, layer: 1, pos: 2031
type: A, layer: 1, pos: 2046
type: B, layer: 1, pos: 2046
type: B, layer: 1, pos: 2645
type: A, layer: 1, pos: 2645
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
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

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2653

## Relational analysis of NS_A2_B1_B2_B2_A1

### Relational analysis result of NS_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146182, upper bound: 0.0145735
time: 133.98 seconds

## Relational analysis of NS_A2_B1_B2_B2_A2

### Relational analysis result of NS_A2_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146229, upper bound: 0.0145740
time: 9.10 seconds

## BFS NS instance: NS_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: 0.9207685, 1.2496402, 0.9206266, 1.2470311, -0.0299662, 0.0324336
1: -1.7842960, -0.8729290, -1.7813430, -0.8731041, -0.3185688, 0.3134242
2: -0.6269078, -0.1481197, -0.6269954, -0.1506746, -0.0477415, 0.0513319
3: -1.4634472, -0.9427649, -1.4626951, -0.9427254, -0.0878159, 0.0850948
4: -1.8172849, -0.9488928, -1.8168143, -0.9513897, -0.1600418, 0.1584009
5: -1.9010626, -1.3564936, -1.9003298, -1.3564854, -0.0889647, 0.0866105
6: -5.1307836, -4.2949815, -5.1290150, -4.2961602, -0.1429961, 0.1383241
7: -2.9596643, -2.1596456, -2.9590900, -2.1595628, -0.1340463, 0.1308623
8: -0.7275938, -0.3988674, -0.7256539, -0.3988752, -0.0322146, 0.0305751
9: -2.6121800, -1.6737008, -2.6078379, -1.6737187, -0.1858795, 0.1813818

Time for backsubstitution: 5.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2653
type: A, layer: 1, pos: 2653
type: B, layer: 1, pos: 2305
type: A, layer: 1, pos: 2305
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 3206
type: A, layer: 1, pos: 3206
type: B, layer: 1, pos: 2058
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 2041
type: A, layer: 1, pos: 2041
type: B, layer: 1, pos: 2141
type: A, layer: 1, pos: 2141
type: B, layer: 1, pos: 2590
type: A, layer: 1, pos: 2590
type: B, layer: 1, pos: 2040
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2026
type: B, layer: 1, pos: 2026
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2042
type: A, layer: 1, pos: 2042
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 2222
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 2025
type: A, layer: 1, pos: 2025
type: B, layer: 1, pos: 3567
type: A, layer: 1, pos: 3102
type: B, layer: 1, pos: 3102
type: B, layer: 1, pos: 2112
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2027
type: B, layer: 1, pos: 2027
type: A, layer: 1, pos: 3459
type: B, layer: 1, pos: 3459
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 2559
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 2043
type: A, layer: 1, pos: 2043
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 2113
type: A, layer: 1, pos: 2113
type: B, layer: 1, pos: 2810
type: A, layer: 1, pos: 2810
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 2098
type: B, layer: 1, pos: 2098
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 2761
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2161
type: B, layer: 1, pos: 2161
type: A, layer: 1, pos: 335
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2127
type: B, layer: 1, pos: 2127
type: A, layer: 1, pos: 2028
type: B, layer: 1, pos: 2028
type: A, layer: 1, pos: 2035
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2063
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 2091
type: A, layer: 1, pos: 2177
type: B, layer: 1, pos: 2177
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2893
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2091
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2207
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2160
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2090
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 3113
type: A, layer: 1, pos: 3113
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2074
type: A, layer: 1, pos: 2762
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2762
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 2090
type: B, layer: 1, pos: 2295
type: A, layer: 1, pos: 2076
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 2059
type: A, layer: 1, pos: 2059
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 3114
type: A, layer: 1, pos: 3114
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2075
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2075
type: B, layer: 1, pos: 2877
type: A, layer: 1, pos: 2877
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 2060
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 3292
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 2176
type: B, layer: 1, pos: 2176
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2200
type: B, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: B, layer: 1, pos: 2424
type: A, layer: 1, pos: 2049
type: B, layer: 1, pos: 2049
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 2034
type: B, layer: 1, pos: 2034
type: A, layer: 1, pos: 2048
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 302
type: A, layer: 1, pos: 2033
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2795
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 2650
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2441
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2646
type: B, layer: 1, pos: 2646
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 2951
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 2201
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2221
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 817
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 2191
type: A, layer: 1, pos: 2202
type: B, layer: 1, pos: 2202
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 2190
type: B, layer: 1, pos: 2190
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 2192
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 2192
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2044
type: B, layer: 1, pos: 2044
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 2217
type: B, layer: 1, pos: 2217
type: A, layer: 1, pos: 2220
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2029
type: B, layer: 1, pos: 2029
type: A, layer: 1, pos: 2616
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 2045
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 2616
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 2030
type: B, layer: 1, pos: 2030
type: A, layer: 1, pos: 2206
type: B, layer: 1, pos: 2206
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 2205
type: B, layer: 1, pos: 2205
type: A, layer: 1, pos: 2218
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2047
type: A, layer: 1, pos: 2047
type: B, layer: 1, pos: 2031
type: A, layer: 1, pos: 2031
type: B, layer: 1, pos: 2046
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2645
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 808
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

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 2653

## Relational analysis of NS_A2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146230, upper bound: 0.0145416
time: 38.97 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146230, upper bound: 0.0145463
time: 153.47 seconds

## BFS NS instance: NS_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: 0.9199223, 1.2498416, 0.9206288, 1.2471914, -0.0309678, 0.0324654
1: -1.7829547, -0.8729485, -1.7802467, -0.8731040, -0.3174284, 0.3125709
2: -0.6298235, -0.1475519, -0.6269934, -0.1502200, -0.0513571, 0.0514195
3: -1.4633722, -0.9431103, -1.4626946, -0.9430079, -0.0881543, 0.0850754
4: -1.8174816, -0.9492509, -1.8168089, -0.9516846, -0.1607358, 0.1583866
5: -1.9008220, -1.3576497, -1.9003303, -1.3574300, -0.0903683, 0.0865634
6: -5.1311917, -4.2912340, -5.1293287, -4.2961621, -0.1431319, 0.1426653
7: -2.9599760, -2.1615226, -2.9590898, -2.1611006, -0.1373867, 0.1307901
8: -0.7280899, -0.3967777, -0.7260476, -0.3988764, -0.0322967, 0.0330533
9: -2.6114256, -1.6733801, -2.6072197, -1.6737256, -0.1858452, 0.1830155

Time for backsubstitution: 5.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2653
type: A, layer: 1, pos: 2653
type: B, layer: 1, pos: 2305
type: A, layer: 1, pos: 2305
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 3206
type: A, layer: 1, pos: 3206
type: B, layer: 1, pos: 2058
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 2041
type: A, layer: 1, pos: 2041
type: B, layer: 1, pos: 2141
type: A, layer: 1, pos: 2141
type: B, layer: 1, pos: 2590
type: A, layer: 1, pos: 2590
type: B, layer: 1, pos: 2040
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2026
type: B, layer: 1, pos: 2026
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2042
type: A, layer: 1, pos: 2042
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 2222
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 2025
type: A, layer: 1, pos: 2025
type: B, layer: 1, pos: 3567
type: A, layer: 1, pos: 3102
type: B, layer: 1, pos: 3102
type: B, layer: 1, pos: 2112
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2027
type: B, layer: 1, pos: 2027
type: A, layer: 1, pos: 3459
type: B, layer: 1, pos: 3459
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 2559
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 2043
type: A, layer: 1, pos: 2043
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 2113
type: A, layer: 1, pos: 2113
type: B, layer: 1, pos: 2810
type: A, layer: 1, pos: 2810
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 2098
type: B, layer: 1, pos: 2098
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 2761
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 2161
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2127
type: B, layer: 1, pos: 2127
type: A, layer: 1, pos: 2028
type: B, layer: 1, pos: 2028
type: A, layer: 1, pos: 2035
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2063
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 2091
type: A, layer: 1, pos: 2177
type: B, layer: 1, pos: 2177
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2893
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2091
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2207
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2160
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2090
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 3113
type: A, layer: 1, pos: 3113
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 2074
type: A, layer: 1, pos: 2762
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2762
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 2090
type: B, layer: 1, pos: 2295
type: A, layer: 1, pos: 2076
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2059
type: A, layer: 1, pos: 2059
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 3114
type: A, layer: 1, pos: 3114
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2075
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2075
type: B, layer: 1, pos: 2877
type: A, layer: 1, pos: 2877
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 2060
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 3292
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 2176
type: B, layer: 1, pos: 2176
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2200
type: B, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: B, layer: 1, pos: 2424
type: A, layer: 1, pos: 2049
type: B, layer: 1, pos: 2049
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 2034
type: B, layer: 1, pos: 2034
type: A, layer: 1, pos: 2048
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 302
type: A, layer: 1, pos: 2033
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2795
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 2650
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2441
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2646
type: B, layer: 1, pos: 2646
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 2951
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 2201
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2221
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 817
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 2191
type: A, layer: 1, pos: 2202
type: B, layer: 1, pos: 2202
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 2190
type: B, layer: 1, pos: 2190
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 2192
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 2192
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2044
type: B, layer: 1, pos: 2044
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 2217
type: B, layer: 1, pos: 2217
type: A, layer: 1, pos: 2220
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2029
type: B, layer: 1, pos: 2029
type: A, layer: 1, pos: 2616
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 2045
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 2616
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 2030
type: B, layer: 1, pos: 2030
type: A, layer: 1, pos: 2206
type: B, layer: 1, pos: 2206
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 2205
type: B, layer: 1, pos: 2205
type: A, layer: 1, pos: 2218
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2047
type: A, layer: 1, pos: 2047
type: B, layer: 1, pos: 2031
type: A, layer: 1, pos: 2031
type: B, layer: 1, pos: 2046
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2645
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 808
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

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2653

## Relational analysis of NS_A2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146230, upper bound: 0.0146034
time: 3.52 seconds

## Relational analysis of NS_A2_B2_A1_A2_B2

### Relational analysis result of NS_A2_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146229, upper bound: 0.0146081
time: 36.05 seconds

## BFS NS instance: NS_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: 0.9206256, 1.2499313, 0.9206264, 1.2472874, -0.0308998, 0.0320301
1: -1.7849746, -0.8731273, -1.7819297, -0.8731041, -0.3167752, 0.3179029
2: -0.6270033, -0.1479619, -0.6269968, -0.1505371, -0.0481480, 0.0511649
3: -1.4635017, -0.9427260, -1.4627433, -0.9427252, -0.0877840, 0.0851896
4: -1.8170476, -0.9488354, -1.8168162, -0.9513363, -0.1614438, 0.1579682
5: -1.9010557, -1.3564768, -1.9003303, -1.3564706, -0.0891012, 0.0865544
6: -5.1308560, -4.2950187, -5.1290798, -4.2961593, -0.1429302, 0.1387339
7: -2.9596062, -2.1595097, -2.9590902, -2.1594439, -0.1352387, 0.1303192
8: -0.7276027, -0.3988442, -0.7256540, -0.3988547, -0.0322457, 0.0305727
9: -2.6122506, -1.6737230, -2.6078992, -1.6737175, -0.1856541, 0.1819543

Time for backsubstitution: 5.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2653
type: A, layer: 1, pos: 2653
type: B, layer: 1, pos: 2305
type: A, layer: 1, pos: 2305
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 3206
type: A, layer: 1, pos: 3206
type: B, layer: 1, pos: 2058
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 2041
type: A, layer: 1, pos: 2041
type: B, layer: 1, pos: 2141
type: A, layer: 1, pos: 2141
type: B, layer: 1, pos: 2590
type: A, layer: 1, pos: 2590
type: B, layer: 1, pos: 2040
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2026
type: B, layer: 1, pos: 2026
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2042
type: A, layer: 1, pos: 2042
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 2222
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 2025
type: A, layer: 1, pos: 2025
type: B, layer: 1, pos: 3567
type: A, layer: 1, pos: 3102
type: B, layer: 1, pos: 3102
type: B, layer: 1, pos: 2112
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2027
type: B, layer: 1, pos: 2027
type: A, layer: 1, pos: 3459
type: B, layer: 1, pos: 3459
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 2559
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 2043
type: A, layer: 1, pos: 2043
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 2113
type: A, layer: 1, pos: 2113
type: B, layer: 1, pos: 2810
type: A, layer: 1, pos: 2810
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 2098
type: B, layer: 1, pos: 2098
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 2761
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 2161
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2127
type: B, layer: 1, pos: 2127
type: A, layer: 1, pos: 2028
type: B, layer: 1, pos: 2028
type: A, layer: 1, pos: 2035
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2063
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 2091
type: A, layer: 1, pos: 2177
type: B, layer: 1, pos: 2177
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2893
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2091
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2207
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2160
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2090
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 3113
type: A, layer: 1, pos: 3113
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2074
type: A, layer: 1, pos: 2762
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2762
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2076
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 3114
type: A, layer: 1, pos: 3114
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2075
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2075
type: B, layer: 1, pos: 2877
type: A, layer: 1, pos: 2877
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 2060
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 3292
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 2176
type: B, layer: 1, pos: 2176
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2200
type: B, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: B, layer: 1, pos: 2424
type: A, layer: 1, pos: 2049
type: B, layer: 1, pos: 2049
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 2034
type: B, layer: 1, pos: 2034
type: A, layer: 1, pos: 2048
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 302
type: A, layer: 1, pos: 2033
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2795
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 2650
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2441
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2646
type: B, layer: 1, pos: 2646
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 2951
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 2201
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2221
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 817
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 2191
type: A, layer: 1, pos: 2202
type: B, layer: 1, pos: 2202
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 2190
type: B, layer: 1, pos: 2190
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 2192
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 2192
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2044
type: B, layer: 1, pos: 2044
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 2217
type: B, layer: 1, pos: 2217
type: A, layer: 1, pos: 2220
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2029
type: B, layer: 1, pos: 2029
type: A, layer: 1, pos: 2616
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 2045
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 2616
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 2030
type: B, layer: 1, pos: 2030
type: A, layer: 1, pos: 2206
type: B, layer: 1, pos: 2206
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 2205
type: B, layer: 1, pos: 2205
type: A, layer: 1, pos: 2218
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2047
type: A, layer: 1, pos: 2047
type: B, layer: 1, pos: 2031
type: A, layer: 1, pos: 2031
type: B, layer: 1, pos: 2046
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2645
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 808
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

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2653

## Relational analysis of NS_A2_B2_A2_A1_B1

### Relational analysis result of NS_A2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146229, upper bound: 0.0145565
time: 3.04 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2

### Relational analysis result of NS_A2_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146230, upper bound: 0.0145613
time: 3.50 seconds

## BFS NS instance: NS_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: 0.9197791, 1.2501332, 0.9206285, 1.2474475, -0.0319014, 0.0320620
1: -1.7836323, -0.8731462, -1.7808332, -0.8731041, -0.3156348, 0.3170501
2: -0.6299191, -0.1473943, -0.6269946, -0.1500825, -0.0517636, 0.0512525
3: -1.4634268, -0.9430708, -1.4627428, -0.9430072, -0.0881224, 0.0851701
4: -1.8172446, -0.9491935, -1.8168118, -0.9516317, -0.1621375, 0.1579541
5: -1.9008145, -1.3576326, -1.9003308, -1.3574156, -0.0905047, 0.0865072
6: -5.1312642, -4.2912703, -5.1293926, -4.2961617, -0.1430660, 0.1430752
7: -2.9599178, -2.1613858, -2.9590898, -2.1609817, -0.1385792, 0.1302468
8: -0.7280990, -0.3967549, -0.7260478, -0.3988560, -0.0323279, 0.0330509
9: -2.6114955, -1.6734028, -2.6072812, -1.6737232, -0.1856200, 0.1835876

Time for backsubstitution: 5.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2653
type: A, layer: 1, pos: 2653
type: B, layer: 1, pos: 2305
type: A, layer: 1, pos: 2305
type: B, layer: 1, pos: 2347
type: A, layer: 1, pos: 2347
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 3206
type: A, layer: 1, pos: 3206
type: B, layer: 1, pos: 2058
type: A, layer: 1, pos: 2058
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 547
type: A, layer: 1, pos: 547
type: B, layer: 1, pos: 2041
type: A, layer: 1, pos: 2041
type: B, layer: 1, pos: 2141
type: A, layer: 1, pos: 2141
type: B, layer: 1, pos: 2590
type: A, layer: 1, pos: 2590
type: B, layer: 1, pos: 2040
type: A, layer: 1, pos: 2040
type: A, layer: 1, pos: 2026
type: B, layer: 1, pos: 2026
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2042
type: A, layer: 1, pos: 2042
type: B, layer: 1, pos: 707
type: A, layer: 1, pos: 707
type: B, layer: 1, pos: 2222
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 534
type: B, layer: 1, pos: 534
type: B, layer: 1, pos: 2539
type: B, layer: 1, pos: 2025
type: A, layer: 1, pos: 2025
type: B, layer: 1, pos: 3567
type: A, layer: 1, pos: 3102
type: B, layer: 1, pos: 3102
type: B, layer: 1, pos: 2112
type: A, layer: 1, pos: 2112
type: A, layer: 1, pos: 2155
type: B, layer: 1, pos: 2155
type: A, layer: 1, pos: 2027
type: B, layer: 1, pos: 2027
type: A, layer: 1, pos: 3459
type: B, layer: 1, pos: 3459
type: A, layer: 1, pos: 2156
type: B, layer: 1, pos: 2156
type: A, layer: 1, pos: 106
type: B, layer: 1, pos: 106
type: A, layer: 1, pos: 2559
type: B, layer: 1, pos: 750
type: A, layer: 1, pos: 750
type: B, layer: 1, pos: 2043
type: A, layer: 1, pos: 2043
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 2113
type: A, layer: 1, pos: 2113
type: B, layer: 1, pos: 2810
type: A, layer: 1, pos: 2810
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 2098
type: B, layer: 1, pos: 2098
type: B, layer: 1, pos: 782
type: A, layer: 1, pos: 782
type: B, layer: 1, pos: 2761
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 335
type: A, layer: 1, pos: 2161
type: B, layer: 1, pos: 2161
type: B, layer: 1, pos: 801
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2127
type: B, layer: 1, pos: 2127
type: A, layer: 1, pos: 2028
type: B, layer: 1, pos: 2028
type: A, layer: 1, pos: 2035
type: B, layer: 1, pos: 2035
type: B, layer: 1, pos: 2063
type: A, layer: 1, pos: 2063
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: B, layer: 1, pos: 2091
type: A, layer: 1, pos: 2177
type: B, layer: 1, pos: 2177
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 692
type: B, layer: 1, pos: 692
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 2893
type: B, layer: 1, pos: 2893
type: B, layer: 1, pos: 815
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 2091
type: B, layer: 1, pos: 736
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 158
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 741
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2207
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2160
type: A, layer: 1, pos: 2160
type: B, layer: 1, pos: 2090
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 3113
type: A, layer: 1, pos: 3113
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 2074
type: A, layer: 1, pos: 2762
type: B, layer: 1, pos: 2076
type: B, layer: 1, pos: 2077
type: B, layer: 1, pos: 2762
type: A, layer: 1, pos: 2074
type: A, layer: 1, pos: 2090
type: A, layer: 1, pos: 2076
type: B, layer: 1, pos: 129
type: A, layer: 1, pos: 2077
type: A, layer: 1, pos: 691
type: B, layer: 1, pos: 691
type: A, layer: 1, pos: 129
type: B, layer: 1, pos: 2295
type: B, layer: 1, pos: 3472
type: B, layer: 1, pos: 2059
type: B, layer: 1, pos: 2061
type: A, layer: 1, pos: 2059
type: A, layer: 1, pos: 2061
type: B, layer: 1, pos: 3114
type: A, layer: 1, pos: 3114
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2075
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2075
type: B, layer: 1, pos: 2877
type: A, layer: 1, pos: 2877
type: B, layer: 1, pos: 2602
type: A, layer: 1, pos: 2602
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 2060
type: A, layer: 1, pos: 2060
type: A, layer: 1, pos: 3292
type: B, layer: 1, pos: 3292
type: B, layer: 1, pos: 816
type: A, layer: 1, pos: 2176
type: B, layer: 1, pos: 2176
type: A, layer: 1, pos: 2175
type: B, layer: 1, pos: 2175
type: A, layer: 1, pos: 816
type: A, layer: 1, pos: 2200
type: B, layer: 1, pos: 2200
type: A, layer: 1, pos: 2424
type: B, layer: 1, pos: 2424
type: A, layer: 1, pos: 2049
type: B, layer: 1, pos: 2049
type: A, layer: 1, pos: 302
type: A, layer: 1, pos: 2034
type: B, layer: 1, pos: 2034
type: A, layer: 1, pos: 2048
type: B, layer: 1, pos: 2048
type: B, layer: 1, pos: 139
type: B, layer: 1, pos: 302
type: A, layer: 1, pos: 2033
type: B, layer: 1, pos: 2033
type: B, layer: 1, pos: 803
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 139
type: A, layer: 1, pos: 2795
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 799
type: A, layer: 1, pos: 799
type: B, layer: 1, pos: 2650
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2441
type: B, layer: 1, pos: 2441
type: A, layer: 1, pos: 2646
type: B, layer: 1, pos: 2646
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: B, layer: 1, pos: 766
type: A, layer: 1, pos: 766
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 2951
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2795
type: B, layer: 1, pos: 2201
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 2221
type: B, layer: 1, pos: 2221
type: B, layer: 1, pos: 817
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 861
type: B, layer: 1, pos: 861
type: B, layer: 1, pos: 2191
type: A, layer: 1, pos: 2202
type: B, layer: 1, pos: 2202
type: A, layer: 1, pos: 821
type: B, layer: 1, pos: 821
type: A, layer: 1, pos: 2190
type: B, layer: 1, pos: 2190
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 2192
type: B, layer: 1, pos: 723
type: A, layer: 1, pos: 723
type: B, layer: 1, pos: 2192
type: A, layer: 1, pos: 868
type: B, layer: 1, pos: 868
type: B, layer: 1, pos: 2421
type: A, layer: 1, pos: 2421
type: A, layer: 1, pos: 2044
type: B, layer: 1, pos: 2044
type: A, layer: 1, pos: 533
type: B, layer: 1, pos: 533
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 2217
type: B, layer: 1, pos: 2217
type: A, layer: 1, pos: 2220
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 156
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2029
type: B, layer: 1, pos: 2029
type: A, layer: 1, pos: 2616
type: B, layer: 1, pos: 511
type: A, layer: 1, pos: 511
type: A, layer: 1, pos: 2045
type: B, layer: 1, pos: 2045
type: B, layer: 1, pos: 2616
type: A, layer: 1, pos: 818
type: B, layer: 1, pos: 818
type: A, layer: 1, pos: 2030
type: B, layer: 1, pos: 2030
type: A, layer: 1, pos: 2206
type: B, layer: 1, pos: 2206
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 2205
type: B, layer: 1, pos: 2205
type: A, layer: 1, pos: 2218
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 2032
type: A, layer: 1, pos: 2032
type: A, layer: 1, pos: 877
type: B, layer: 1, pos: 877
type: B, layer: 1, pos: 2047
type: A, layer: 1, pos: 2047
type: B, layer: 1, pos: 2031
type: A, layer: 1, pos: 2031
type: B, layer: 1, pos: 2046
type: A, layer: 1, pos: 2046
type: A, layer: 1, pos: 2645
type: B, layer: 1, pos: 2645
type: B, layer: 1, pos: 2145
type: A, layer: 1, pos: 2145
type: B, layer: 1, pos: 808
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

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2653

## Relational analysis of NS_A2_B2_A2_A2_B1

### Relational analysis result of NS_A2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146229, upper bound: 0.0146183
time: 28.24 seconds

## Relational analysis of NS_A2_B2_A2_A2_B2

### Relational analysis result of NS_A2_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146230, upper bound: 0.0146230
time: 3.54 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 37.75 seconds
NS_A1_B2_A2_A2_B1, status: Status.VERIFIED, split count: 5, time: 37.75
Output dim: 0, lower bound: -0.0144853, upper bound: 0.0146182
NS_A1_B2_A2_A2_B2, status: Status.VERIFIED, split count: 5, time: 37.75
Output dim: 0, lower bound: -0.0144853, upper bound: 0.0146230
NS_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 37.75
Output dim: 0, lower bound: -0.0146182, upper bound: 0.0145735
NS_A2_B1_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 37.75
Output dim: 0, lower bound: -0.0146229, upper bound: 0.0145740
NS_A2_B2_A1_A1_B1, status: Status.VERIFIED, split count: 5, time: 37.75
Output dim: 0, lower bound: -0.0146230, upper bound: 0.0145416
NS_A2_B2_A1_A1_B2, status: Status.VERIFIED, split count: 5, time: 37.75
Output dim: 0, lower bound: -0.0146230, upper bound: 0.0145463
NS_A2_B2_A1_A2_B1, status: Status.VERIFIED, split count: 5, time: 37.75
Output dim: 0, lower bound: -0.0146230, upper bound: 0.0146034
NS_A2_B2_A1_A2_B2, status: Status.VERIFIED, split count: 5, time: 37.75
Output dim: 0, lower bound: -0.0146229, upper bound: 0.0146081
NS_A2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 5, time: 37.75
Output dim: 0, lower bound: -0.0146229, upper bound: 0.0145565
NS_A2_B2_A2_A1_B2, status: Status.VERIFIED, split count: 5, time: 37.75
Output dim: 0, lower bound: -0.0146230, upper bound: 0.0145613
NS_A2_B2_A2_A2_B1, status: Status.VERIFIED, split count: 5, time: 37.75
Output dim: 0, lower bound: -0.0146229, upper bound: 0.0146183
NS_A2_B2_A2_A2_B2, status: Status.VERIFIED, split count: 5, time: 37.75
Output dim: 0, lower bound: -0.0146230, upper bound: 0.0146230

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 32.63 + 1598.30 = 1630.92 seconds
