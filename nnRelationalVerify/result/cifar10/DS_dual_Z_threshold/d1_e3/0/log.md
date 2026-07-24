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
execution time: IAR + RelationalAnalysis = 9.63 + 24.70 = 34.32 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0146387, upper bound: 0.0146385

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2206

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146384, upper bound: 0.0146384
time: 3.24 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146384, upper bound: 0.0146384
time: 3.21 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.52 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.52
Output dim: 0, lower bound: -0.0146384, upper bound: 0.0146384
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.52
Output dim: 0, lower bound: -0.0146384, upper bound: 0.0146384

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317527, 0.0317527
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3185104, 0.3185105
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496898, 0.0496898
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0872871, 0.0872871
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611761, 0.1611761
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0887313, 0.0887313
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1424296, 0.1424297
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1352778, 0.1352779
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314895, 0.0314895
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1832749, 0.1832749

Time for backsubstitution: 7.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2205

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146374, upper bound: 0.0146380
time: 3.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146379, upper bound: 0.0146373
time: 69.28 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317527, 0.0317527
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3185104, 0.3185105
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496898, 0.0496898
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0872871, 0.0872871
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611761, 0.1611761
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0887313, 0.0887313
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1424296, 0.1424297
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1352778, 0.1352779
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314895, 0.0314895
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1832749, 0.1832749

Time for backsubstitution: 5.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2205

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146374, upper bound: 0.0146380
time: 3.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146379, upper bound: 0.0146375
time: 4.92 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 14.21 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 14.21
Output dim: 0, lower bound: -0.0146374, upper bound: 0.0146380
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 14.21
Output dim: 0, lower bound: -0.0146379, upper bound: 0.0146373
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 14.21
Output dim: 0, lower bound: -0.0146374, upper bound: 0.0146380
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 14.21
Output dim: 0, lower bound: -0.0146379, upper bound: 0.0146375

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317526, 0.0317525
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3185103, 0.3185103
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496898, 0.0496898
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0872869, 0.0872869
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611755, 0.1611755
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0887309, 0.0887309
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1424296, 0.1424296
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1352770, 0.1352770
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314890, 0.0314890
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1832743, 0.1832743

Time for backsubstitution: 5.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2221

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146362, upper bound: 0.0146370
time: 152.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146367, upper bound: 0.0146369
time: 4.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317526, 0.0317525
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3185103, 0.3185103
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496898, 0.0496898
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0872869, 0.0872869
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611755, 0.1611755
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0887309, 0.0887309
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1424296, 0.1424296
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1352770, 0.1352770
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314890, 0.0314890
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1832743, 0.1832743

Time for backsubstitution: 5.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2221

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146368, upper bound: 0.0146368
time: 4.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146372, upper bound: 0.0146361
time: 53.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317525, 0.0317526
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3185103, 0.3185103
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496898, 0.0496898
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0872869, 0.0872869
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611755, 0.1611755
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0887309, 0.0887309
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1424296, 0.1424296
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1352770, 0.1352770
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314890, 0.0314890
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1832743, 0.1832743

Time for backsubstitution: 5.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2221

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146362, upper bound: 0.0146369
time: 17.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146366, upper bound: 0.0146365
time: 2.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317525, 0.0317526
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3185103, 0.3185103
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496898, 0.0496898
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0872869, 0.0872869
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611755, 0.1611755
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0887309, 0.0887309
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1424296, 0.1424296
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1352770, 0.1352769
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314890, 0.0314890
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1832743, 0.1832743

Time for backsubstitution: 6.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2221

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146367, upper bound: 0.0146367
time: 43.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146372, upper bound: 0.0146361
time: 71.45 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 120.83 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 120.83
Output dim: 0, lower bound: -0.0146362, upper bound: 0.0146370
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 120.83
Output dim: 0, lower bound: -0.0146367, upper bound: 0.0146369
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 120.83
Output dim: 0, lower bound: -0.0146368, upper bound: 0.0146368
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 120.83
Output dim: 0, lower bound: -0.0146372, upper bound: 0.0146361
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 120.83
Output dim: 0, lower bound: -0.0146362, upper bound: 0.0146369
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 120.83
Output dim: 0, lower bound: -0.0146366, upper bound: 0.0146365
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 120.83
Output dim: 0, lower bound: -0.0146367, upper bound: 0.0146367
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 120.83
Output dim: 0, lower bound: -0.0146372, upper bound: 0.0146361

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317401, 0.0317400
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3184955, 0.3184958
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496800, 0.0496803
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0872832, 0.0872832
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611642, 0.1611654
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0887301, 0.0887302
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1424161, 0.1424160
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1352638, 0.1352648
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314885, 0.0314885
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1832542, 0.1832536

Time for backsubstitution: 6.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2191

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146354, upper bound: 0.0146346
time: 24.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146336, upper bound: 0.0146365
time: 7.77 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317400, 0.0317401
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3184958, 0.3184954
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496803, 0.0496800
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0872832, 0.0872832
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611653, 0.1611643
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0887302, 0.0887301
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1424160, 0.1424161
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1352647, 0.1352638
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314885, 0.0314885
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1832537, 0.1832541

Time for backsubstitution: 6.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2191

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146359, upper bound: 0.0146342
time: 79.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146341, upper bound: 0.0146361
time: 18.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317401, 0.0317400
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3184954, 0.3184958
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496800, 0.0496803
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0872832, 0.0872832
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611643, 0.1611654
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0887301, 0.0887302
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1424161, 0.1424160
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1352638, 0.1352648
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314885, 0.0314885
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1832542, 0.1832537

Time for backsubstitution: 6.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2191

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146360, upper bound: 0.0146341
time: 3.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146342, upper bound: 0.0146358
time: 67.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317400, 0.0317401
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3184958, 0.3184955
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496803, 0.0496800
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0872832, 0.0872832
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611654, 0.1611643
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0887302, 0.0887301
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1424160, 0.1424161
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1352647, 0.1352638
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314885, 0.0314885
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1832537, 0.1832542

Time for backsubstitution: 6.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2191

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146365, upper bound: 0.0146337
time: 3.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146346, upper bound: 0.0146355
time: 3.29 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317401, 0.0317400
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3184955, 0.3184958
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496800, 0.0496803
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0872832, 0.0872832
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611643, 0.1611653
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0887301, 0.0887302
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1424161, 0.1424160
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1352638, 0.1352648
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314885, 0.0314885
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1832542, 0.1832537

Time for backsubstitution: 6.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2191

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146354, upper bound: 0.0146345
time: 3.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146336, upper bound: 0.0146365
time: 9.51 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317400, 0.0317401
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3184958, 0.3184954
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496803, 0.0496800
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0872832, 0.0872832
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611654, 0.1611643
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0887302, 0.0887301
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1424160, 0.1424161
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1352648, 0.1352638
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314885, 0.0314885
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1832537, 0.1832542

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2191

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146359, upper bound: 0.0146343
time: 3.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146341, upper bound: 0.0146361
time: 3.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317401, 0.0317400
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3184954, 0.3184958
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496800, 0.0496803
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0872832, 0.0872832
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611643, 0.1611653
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0887301, 0.0887302
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1424161, 0.1424160
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1352638, 0.1352648
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314885, 0.0314885
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1832541, 0.1832538

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2191

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146360, upper bound: 0.0146340
time: 83.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146341, upper bound: 0.0146361
time: 3.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317400, 0.0317401
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3184958, 0.3184955
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496803, 0.0496800
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0872832, 0.0872832
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611654, 0.1611642
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0887302, 0.0887301
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1424160, 0.1424161
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1352648, 0.1352638
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314885, 0.0314885
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1832536, 0.1832542

Time for backsubstitution: 6.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2191

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146364, upper bound: 0.0146336
time: 3.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146346, upper bound: 0.0146351
time: 512.03 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 522.28 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 522.28
Output dim: 0, lower bound: -0.0146354, upper bound: 0.0146346
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 522.28
Output dim: 0, lower bound: -0.0146336, upper bound: 0.0146365
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 522.28
Output dim: 0, lower bound: -0.0146359, upper bound: 0.0146342
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 522.28
Output dim: 0, lower bound: -0.0146341, upper bound: 0.0146361
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 522.28
Output dim: 0, lower bound: -0.0146360, upper bound: 0.0146341
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 522.28
Output dim: 0, lower bound: -0.0146342, upper bound: 0.0146358
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 522.28
Output dim: 0, lower bound: -0.0146365, upper bound: 0.0146337
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 522.28
Output dim: 0, lower bound: -0.0146346, upper bound: 0.0146355
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 522.28
Output dim: 0, lower bound: -0.0146354, upper bound: 0.0146345
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 522.28
Output dim: 0, lower bound: -0.0146336, upper bound: 0.0146365
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 522.28
Output dim: 0, lower bound: -0.0146359, upper bound: 0.0146343
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 522.28
Output dim: 0, lower bound: -0.0146341, upper bound: 0.0146361
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 522.28
Output dim: 0, lower bound: -0.0146360, upper bound: 0.0146340
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 522.28
Output dim: 0, lower bound: -0.0146341, upper bound: 0.0146361
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 522.28
Output dim: 0, lower bound: -0.0146364, upper bound: 0.0146336
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 522.28
Output dim: 0, lower bound: -0.0146346, upper bound: 0.0146351

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317332, 0.0317327
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3184586, 0.3184555
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496467, 0.0496443
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0872807, 0.0872806
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611445, 0.1611477
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0887249, 0.0887253
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1423681, 0.1423678
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1352501, 0.1352521
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314725, 0.0314737
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1831348, 0.1831204

Time for backsubstitution: 6.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146333, upper bound: 0.0146321
time: 27.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146343, upper bound: 0.0146310
time: 5.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317329, 0.0317331
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3184553, 0.3184588
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496440, 0.0496469
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0872807, 0.0872807
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611466, 0.1611456
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0887252, 0.0887249
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1423679, 0.1423681
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1352511, 0.1352511
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314737, 0.0314725
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1831210, 0.1831343

Time for backsubstitution: 6.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146298, upper bound: 0.0146353
time: 111.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146323, upper bound: 0.0146336
time: 64.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317331, 0.0317328
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3184589, 0.3184552
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496470, 0.0496440
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0872807, 0.0872807
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611456, 0.1611467
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0887249, 0.0887252
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1423681, 0.1423679
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1352510, 0.1352511
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314725, 0.0314737
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1831344, 0.1831208

Time for backsubstitution: 6.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146338, upper bound: 0.0146328
time: 18.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146348, upper bound: 0.0146305
time: 24.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317327, 0.0317332
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3184555, 0.3184584
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496443, 0.0496467
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0872806, 0.0872807
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611477, 0.1611445
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0887253, 0.0887249
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1423678, 0.1423682
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1352521, 0.1352501
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314737, 0.0314725
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1831205, 0.1831347

Time for backsubstitution: 6.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146303, upper bound: 0.0146350
time: 5.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146328, upper bound: 0.0146340
time: 5.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317332, 0.0317327
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3184585, 0.3184555
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496467, 0.0496443
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0872807, 0.0872806
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611445, 0.1611477
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0887249, 0.0887253
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1423681, 0.1423678
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1352501, 0.1352521
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314725, 0.0314737
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1831348, 0.1831204

Time for backsubstitution: 6.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2190

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146340, upper bound: 0.0146329
time: 16.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146349, upper bound: 0.0146302
time: 120.59 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 143.99 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 143.99
Output dim: 0, lower bound: -0.0146333, upper bound: 0.0146321
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 143.99
Output dim: 0, lower bound: -0.0146343, upper bound: 0.0146310
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 143.99
Output dim: 0, lower bound: -0.0146298, upper bound: 0.0146353
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 143.99
Output dim: 0, lower bound: -0.0146323, upper bound: 0.0146336
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 143.99
Output dim: 0, lower bound: -0.0146338, upper bound: 0.0146328
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 143.99
Output dim: 0, lower bound: -0.0146348, upper bound: 0.0146305
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 143.99
Output dim: 0, lower bound: -0.0146303, upper bound: 0.0146350
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 143.99
Output dim: 0, lower bound: -0.0146328, upper bound: 0.0146340
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 143.99
Output dim: 0, lower bound: -0.0146340, upper bound: 0.0146329
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 143.99
Output dim: 0, lower bound: -0.0146349, upper bound: 0.0146302
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 143.99
Output dim: 0, lower bound: -0.0146342, upper bound: 0.0146358
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 143.99
Output dim: 0, lower bound: -0.0146365, upper bound: 0.0146337
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 143.99
Output dim: 0, lower bound: -0.0146346, upper bound: 0.0146355
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 143.99
Output dim: 0, lower bound: -0.0146354, upper bound: 0.0146345
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 143.99
Output dim: 0, lower bound: -0.0146336, upper bound: 0.0146365
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 143.99
Output dim: 0, lower bound: -0.0146359, upper bound: 0.0146343
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 143.99
Output dim: 0, lower bound: -0.0146341, upper bound: 0.0146361
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 143.99
Output dim: 0, lower bound: -0.0146360, upper bound: 0.0146340
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 143.99
Output dim: 0, lower bound: -0.0146341, upper bound: 0.0146361
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 143.99
Output dim: 0, lower bound: -0.0146364, upper bound: 0.0146336
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 143.99
Output dim: 0, lower bound: -0.0146346, upper bound: 0.0146351

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 34.32 + 1787.69 = 1822.01 seconds
