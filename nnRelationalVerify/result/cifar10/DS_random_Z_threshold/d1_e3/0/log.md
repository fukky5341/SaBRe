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
execution time: IAR + RelationalAnalysis = 7.88 + 24.84 = 32.72 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0146387, upper bound: 0.0146385

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2155

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2032

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146382, upper bound: 0.0146381
time: 11.03 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146381, upper bound: 0.0146384
time: 9.95 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 20.99 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 20.99
Output dim: 0, lower bound: -0.0146382, upper bound: 0.0146381
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 20.99
Output dim: 0, lower bound: -0.0146381, upper bound: 0.0146384

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317527, 0.0317527
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3185107, 0.3185106
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496899, 0.0496899
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0872872, 0.0872872
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611760, 0.1611760
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0887313, 0.0887313
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1424294, 0.1424294
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1352778, 0.1352778
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314895, 0.0314895
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1832750, 0.1832750

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2145

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2177

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146376, upper bound: 0.0146305
time: 102.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146307, upper bound: 0.0146375
time: 11.26 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317527, 0.0317527
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3185107, 0.3185106
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496899, 0.0496899
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0872872, 0.0872872
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611760, 0.1611760
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0887313, 0.0887313
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1424295, 0.1424294
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1352778, 0.1352778
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314895, 0.0314895
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1832750, 0.1832750

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2127

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146379, upper bound: 0.0146237
time: 3.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146235, upper bound: 0.0146380
time: 11.64 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 21.94 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 21.94
Output dim: 0, lower bound: -0.0146376, upper bound: 0.0146305
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 21.94
Output dim: 0, lower bound: -0.0146307, upper bound: 0.0146375
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 21.94
Output dim: 0, lower bound: -0.0146379, upper bound: 0.0146237
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 21.94
Output dim: 0, lower bound: -0.0146235, upper bound: 0.0146380

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0316582, 0.0316568
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3153886, 0.3153931
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0494727, 0.0494727
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0859662, 0.0859940
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1601548, 0.1601648
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0870896, 0.0871492
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1422558, 0.1422590
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1322976, 0.1323293
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0313597, 0.0313596
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1786773, 0.1786836

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2025

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2099

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146377, upper bound: 0.0146307
time: 3.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146377, upper bound: 0.0146307
time: 3.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0316568, 0.0316582
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3153931, 0.3153887
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0494727, 0.0494727
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0859940, 0.0859662
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1601648, 0.1601548
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0871492, 0.0870896
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1422589, 0.1422558
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1323293, 0.1322976
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0313596, 0.0313597
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1786836, 0.1786773

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2041

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2027

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146301, upper bound: 0.0146180
time: 3.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146109, upper bound: 0.0146371
time: 3.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317191, 0.0317106
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3183338, 0.3182890
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496795, 0.0496803
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0870610, 0.0871066
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611602, 0.1611626
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0884928, 0.0885410
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1423107, 0.1423344
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1349794, 0.1350395
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314795, 0.0314770
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1830946, 0.1830407

Time for backsubstitution: 6.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2027

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2441

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146373, upper bound: 0.0146226
time: 36.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146373, upper bound: 0.0146231
time: 25.30 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317106, 0.0317191
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3182890, 0.3183338
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496803, 0.0496795
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0871066, 0.0870610
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611625, 0.1611602
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0885410, 0.0884929
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1423344, 0.1423107
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1350395, 0.1349794
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314770, 0.0314795
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1830407, 0.1830946

Time for backsubstitution: 6.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 692

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2091

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146174, upper bound: 0.0146300
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146156, upper bound: 0.0146318
time: 3.28 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 12.76 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 12.76
Output dim: 0, lower bound: -0.0146377, upper bound: 0.0146307
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 12.76
Output dim: 0, lower bound: -0.0146377, upper bound: 0.0146307
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 12.76
Output dim: 0, lower bound: -0.0146301, upper bound: 0.0146180
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 12.76
Output dim: 0, lower bound: -0.0146109, upper bound: 0.0146371
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 12.76
Output dim: 0, lower bound: -0.0146373, upper bound: 0.0146226
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 12.76
Output dim: 0, lower bound: -0.0146373, upper bound: 0.0146231
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 12.76
Output dim: 0, lower bound: -0.0146174, upper bound: 0.0146300
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 12.76
Output dim: 0, lower bound: -0.0146156, upper bound: 0.0146318

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0316582, 0.0316568
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3153886, 0.3153931
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0494727, 0.0494727
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0859662, 0.0859940
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1601548, 0.1601648
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0870896, 0.0871492
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1422558, 0.1422590
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1322976, 0.1323293
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0313597, 0.0313596
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1786773, 0.1786836

Time for backsubstitution: 6.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2049

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2060

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146353, upper bound: 0.0146283
time: 48.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146353, upper bound: 0.0146284
time: 3.97 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0316582, 0.0316568
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3153886, 0.3153931
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0494727, 0.0494727
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0859662, 0.0859940
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1601548, 0.1601648
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0870896, 0.0871492
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1422558, 0.1422590
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1322976, 0.1323293
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0313597, 0.0313596
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1786773, 0.1786836

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 892

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 61

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146374, upper bound: 0.0145965
time: 79.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146035, upper bound: 0.0146304
time: 26.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0316059, 0.0315967
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3151458, 0.3151774
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0494446, 0.0494445
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0859870, 0.0859604
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1600860, 0.1600617
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0871394, 0.0870777
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1422092, 0.1422136
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1322532, 0.1322068
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0313566, 0.0313561
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1786409, 0.1786357

Time for backsubstitution: 6.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 808

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 764

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146302, upper bound: 0.0146108
time: 41.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146302, upper bound: 0.0146176
time: 60.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0315953, 0.0316073
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3151819, 0.3151413
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0494446, 0.0494446
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0859882, 0.0859593
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1600717, 0.1600761
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0871373, 0.0870798
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1422167, 0.1422060
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1322385, 0.1322215
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0313560, 0.0313567
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1786420, 0.1786345

Time for backsubstitution: 6.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2650

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2602

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146099, upper bound: 0.0146357
time: 102.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146097, upper bound: 0.0146360
time: 2.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317042, 0.0316956
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3163978, 0.3162163
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0495818, 0.0495852
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0864702, 0.0865351
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1594312, 0.1595645
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0877764, 0.0878596
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1421624, 0.1421882
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1335660, 0.1337047
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0312943, 0.0312903
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1806256, 0.1803990

Time for backsubstitution: 6.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 778

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2145

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146368, upper bound: 0.0146179
time: 4.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146324, upper bound: 0.0146222
time: 7.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0317041, 0.0316957
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3162612, 0.3163531
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0495843, 0.0495827
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0864895, 0.0865158
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1595621, 0.1594336
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0878115, 0.0878246
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1421646, 0.1421861
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1336446, 0.1336262
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0312928, 0.0312918
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1804529, 0.1805717

Time for backsubstitution: 6.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2239

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2090

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146349, upper bound: 0.0146123
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146270, upper bound: 0.0146205
time: 3.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0316950, 0.0317029
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3181578, 0.3180876
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496529, 0.0496719
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0868852, 0.0869398
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611311, 0.1611587
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0883284, 0.0883813
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1420339, 0.1421397
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1347995, 0.1348473
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314769, 0.0314794
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1828692, 0.1828513

Time for backsubstitution: 6.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 137

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146171, upper bound: 0.0146186
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146061, upper bound: 0.0146299
time: 2.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0316944, 0.0317035
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3180428, 0.3182025
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0496727, 0.0496520
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0869854, 0.0868395
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1611611, 0.1611287
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0884295, 0.0882802
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1421634, 0.1420102
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1349074, 0.1347394
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0314769, 0.0314794
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1827974, 0.1829231

Time for backsubstitution: 6.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2077

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146140, upper bound: 0.0146231
time: 156.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146069, upper bound: 0.0146300
time: 18.23 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 181.53 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 181.53
Output dim: 0, lower bound: -0.0146353, upper bound: 0.0146283
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 181.53
Output dim: 0, lower bound: -0.0146353, upper bound: 0.0146284
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 181.53
Output dim: 0, lower bound: -0.0146374, upper bound: 0.0145965
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 181.53
Output dim: 0, lower bound: -0.0146035, upper bound: 0.0146304
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 181.53
Output dim: 0, lower bound: -0.0146302, upper bound: 0.0146108
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 181.53
Output dim: 0, lower bound: -0.0146302, upper bound: 0.0146176
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 181.53
Output dim: 0, lower bound: -0.0146099, upper bound: 0.0146357
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 181.53
Output dim: 0, lower bound: -0.0146097, upper bound: 0.0146360
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 181.53
Output dim: 0, lower bound: -0.0146368, upper bound: 0.0146179
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 181.53
Output dim: 0, lower bound: -0.0146324, upper bound: 0.0146222
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 181.53
Output dim: 0, lower bound: -0.0146349, upper bound: 0.0146123
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 181.53
Output dim: 0, lower bound: -0.0146270, upper bound: 0.0146205
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 181.53
Output dim: 0, lower bound: -0.0146171, upper bound: 0.0146186
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 181.53
Output dim: 0, lower bound: -0.0146061, upper bound: 0.0146299
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 181.53
Output dim: 0, lower bound: -0.0146140, upper bound: 0.0146231
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 181.53
Output dim: 0, lower bound: -0.0146069, upper bound: 0.0146300

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0316463, 0.0316450
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3153189, 0.3153198
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0494679, 0.0494679
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0859160, 0.0859530
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1601545, 0.1601644
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0870413, 0.0871012
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1422471, 0.1422503
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1322490, 0.1322806
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0313554, 0.0313553
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1785883, 0.1785921

Time for backsubstitution: 6.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3567

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2046

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146345, upper bound: 0.0146280
time: 74.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146349, upper bound: 0.0146277
time: 25.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0316464, 0.0316450
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3153153, 0.3153234
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0494679, 0.0494679
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0859253, 0.0859438
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1601545, 0.1601644
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0870416, 0.0871009
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1422471, 0.1422503
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1322489, 0.1322807
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0313554, 0.0313553
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1785858, 0.1785947

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2424

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2218

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146352, upper bound: 0.0146281
time: 6.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146351, upper bound: 0.0146284
time: 7.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0312988, 0.0312852
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3135470, 0.3135813
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0493859, 0.0493863
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0859503, 0.0859787
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1593300, 0.1593051
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0870882, 0.0871479
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1422014, 0.1422053
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1320891, 0.1321214
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0313347, 0.0313340
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1784910, 0.1784953

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2834

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2761

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146324, upper bound: 0.0145929
time: 4.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146336, upper bound: 0.0145919
time: 3.03 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0312866, 0.0312974
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3135768, 0.3135515
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0493862, 0.0493860
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0859509, 0.0859781
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1592951, 0.1593400
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0870883, 0.0871478
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1422021, 0.1422047
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1320897, 0.1321208
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0313341, 0.0313346
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1784889, 0.1784974

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2043

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2202

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146029, upper bound: 0.0146212
time: 5.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0145943, upper bound: 0.0146300
time: 33.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0316059, 0.0315967
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3151458, 0.3151774
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0494446, 0.0494445
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0859870, 0.0859604
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1600860, 0.1600617
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0871394, 0.0870777
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1422092, 0.1422136
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1322532, 0.1322068
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0313566, 0.0313561
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1786409, 0.1786357

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2045

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2075

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146284, upper bound: 0.0146151
time: 8.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146275, upper bound: 0.0146161
time: 20.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0316059, 0.0315967
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3151458, 0.3151774
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0494446, 0.0494445
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0859870, 0.0859604
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1600860, 0.1600617
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0871394, 0.0870777
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1422092, 0.1422136
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1322532, 0.1322068
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0313566, 0.0313561
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1786409, 0.1786357

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2045

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2127

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0146029
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146155, upper bound: 0.0146176
time: 2.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0315517, 0.0315634
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3131981, 0.3131701
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0493150, 0.0493152
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0830939, 0.0830672
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1553721, 0.1553478
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0842038, 0.0841377
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1411495, 0.1411527
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1288841, 0.1288483
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0311990, 0.0311998
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1759077, 0.1759184

Time for backsubstitution: 6.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2127

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146098, upper bound: 0.0146212
time: 6.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0145953, upper bound: 0.0146357
time: 3.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0315515, 0.0315636
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3132107, 0.3131573
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0493152, 0.0493150
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0830962, 0.0830649
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1553434, 0.1553765
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0841952, 0.0841463
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1411633, 0.1411388
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1288654, 0.1288670
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0311991, 0.0311997
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1759259, 0.1759003

Time for backsubstitution: 6.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2040

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 779

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146097, upper bound: 0.0146361
time: 4.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146097, upper bound: 0.0146362
time: 5.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0316433, 0.0316325
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3161470, 0.3159657
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0493810, 0.0493794
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0864626, 0.0865274
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1593404, 0.1594736
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0877430, 0.0878266
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1414756, 0.1415157
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1335318, 0.1336705
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0312392, 0.0312352
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1796281, 0.1794009

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 740

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 817

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146358, upper bound: 0.0146170
time: 5.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146358, upper bound: 0.0146168
time: 6.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0316412, 0.0316347
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3161472, 0.3159655
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0493761, 0.0493844
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0864626, 0.0865274
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1593403, 0.1594737
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0877434, 0.0878262
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1414900, 0.1415013
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1335318, 0.1336705
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0312392, 0.0312352
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1796275, 0.1794015

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2191

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 764

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146323, upper bound: 0.0146223
time: 3.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146323, upper bound: 0.0146222
time: 2.96 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0316930, 0.0316806
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3161319, 0.3161803
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0495737, 0.0495752
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0863215, 0.0864227
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1595515, 0.1594311
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0876507, 0.0877379
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1420680, 0.1421287
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1334648, 0.1335278
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0312927, 0.0312918
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1802884, 0.1803865

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2048

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2616

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146344, upper bound: 0.0146125
time: 13.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146348, upper bound: 0.0146121
time: 100.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0316891, 0.0316846
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3160883, 0.3162239
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0495767, 0.0495721
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0863964, 0.0863478
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1595596, 0.1594229
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0877248, 0.0876637
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1421071, 0.1420896
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1335462, 0.1334464
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0312927, 0.0312917
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1802677, 0.1804072

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146254, upper bound: 0.0146116
time: 3.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146184, upper bound: 0.0146187
time: 15.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0315123, 0.0315283
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3152320, 0.3151500
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0494116, 0.0494336
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0841878, 0.0841767
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1604829, 0.1605126
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0852920, 0.0852610
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1417534, 0.1418368
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1305607, 0.1306038
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0313054, 0.0313067
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1778182, 0.1777843

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2205

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 740

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0146042, upper bound: 0.0146022
time: 133.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146041, upper bound: 0.0146279
time: 3.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0316422, 0.0316516
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3160507, 0.3162026
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0494896, 0.0494696
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0866282, 0.0864677
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1599825, 0.1599523
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0878961, 0.0877256
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1421614, 0.1420083
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1333234, 0.1331611
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0313848, 0.0313870
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1802658, 0.1803817

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 861

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2191

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146062, upper bound: 0.0146277
time: 6.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146044, upper bound: 0.0146294
time: 17.52 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 30.14 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146345, upper bound: 0.0146280
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146349, upper bound: 0.0146277
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146352, upper bound: 0.0146281
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146351, upper bound: 0.0146284
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146324, upper bound: 0.0145929
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146336, upper bound: 0.0145919
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146029, upper bound: 0.0146212
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0145943, upper bound: 0.0146300
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146284, upper bound: 0.0146151
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146275, upper bound: 0.0146161
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0146029
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146155, upper bound: 0.0146176
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146098, upper bound: 0.0146212
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0145953, upper bound: 0.0146357
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146097, upper bound: 0.0146361
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146097, upper bound: 0.0146362
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146358, upper bound: 0.0146170
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146358, upper bound: 0.0146168
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146323, upper bound: 0.0146223
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146323, upper bound: 0.0146222
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146344, upper bound: 0.0146125
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146348, upper bound: 0.0146121
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146254, upper bound: 0.0146116
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146184, upper bound: 0.0146187
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146042, upper bound: 0.0146022
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146041, upper bound: 0.0146279
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146062, upper bound: 0.0146277
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.14
Output dim: 0, lower bound: -0.0146044, upper bound: 0.0146294

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0316455, 0.0316442
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3153188, 0.3153198
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0494677, 0.0494678
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0859157, 0.0859527
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1601532, 0.1601632
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0870411, 0.0871011
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1422457, 0.1422489
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1322485, 0.1322800
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0313551, 0.0313550
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1785882, 0.1785920

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 137

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 692

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146327, upper bound: 0.0146230
time: 7.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146296, upper bound: 0.0146260
time: 44.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0316455, 0.0316442
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3153188, 0.3153198
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0494677, 0.0494678
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0859157, 0.0859527
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1601532, 0.1601632
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0870411, 0.0871011
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1422457, 0.1422489
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1322485, 0.1322800
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0313551, 0.0313550
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1785882, 0.1785920

Time for backsubstitution: 6.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2090

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2155

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146296, upper bound: 0.0146222
time: 13.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146295, upper bound: 0.0146222
time: 72.25 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0316450, 0.0316436
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3149613, 0.3149457
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0494676, 0.0494677
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0859251, 0.0859437
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1599534, 0.1599755
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0870142, 0.0870751
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1422209, 0.1422245
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1320871, 0.1321302
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0313232, 0.0313227
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1781129, 0.1780905

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 740

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 892

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146352, upper bound: 0.0146282
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146352, upper bound: 0.0146282
time: 3.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0316450, 0.0316436
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3149376, 0.3149694
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0494676, 0.0494677
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0859251, 0.0859437
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1599655, 0.1599634
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0870159, 0.0870734
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1422214, 0.1422240
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1320984, 0.1321189
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0313228, 0.0313231
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1780816, 0.1781219

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2762

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 799

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146317, upper bound: 0.0146246
time: 5.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146316, upper bound: 0.0146248
time: 6.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0312612, 0.0312505
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3134257, 0.3134577
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0493833, 0.0493836
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0859490, 0.0859768
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1592275, 0.1592082
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0870861, 0.0871456
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1421919, 0.1421956
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1320716, 0.1321042
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0313246, 0.0313245
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1784848, 0.1784894

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2042

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 547

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146324, upper bound: 0.0144942
time: 4.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145339, upper bound: 0.0145927
time: 25.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0312641, 0.0312474
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3134235, 0.3134601
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0493833, 0.0493836
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0859484, 0.0859773
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1592331, 0.1592026
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0870859, 0.0871458
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1421918, 0.1421957
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1320719, 0.1321039
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0313253, 0.0313239
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1784852, 0.1784891

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 779

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 707

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146290, upper bound: 0.0145847
time: 21.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0146270, upper bound: 0.0145867
time: 7.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9206190, 1.2475445, 0.9206190, 1.2475445, -0.0312833, 0.0312947
1: -1.7828486, -0.8728766, -1.7828486, -0.8728766, -0.3126101, 0.3126652
2: -0.6270034, -0.1499570, -0.6270034, -0.1499570, -0.0492160, 0.0492157
3: -1.4627892, -0.9427141, -1.4627892, -0.9427141, -0.0855583, 0.0855584
4: -1.8168863, -0.9508913, -1.8168863, -0.9508913, -0.1590131, 0.1590286
5: -1.9003969, -1.3564321, -1.9003969, -1.3564321, -0.0866463, 0.0866663
6: -5.1299973, -4.2961111, -5.1299973, -4.2961111, -0.1421987, 0.1422016
7: -2.9591417, -2.1590416, -2.9591417, -2.1590416, -0.1313130, 0.1312707
8: -0.7260501, -0.3987779, -0.7260501, -0.3987779, -0.0312332, 0.0312315
9: -2.6081173, -1.6733506, -2.6081173, -1.6733506, -0.1772099, 0.1773291

Time for backsubstitution: 6.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3567
type: DSZ, layer: 1, pos: 2645
type: DSZ, layer: 1, pos: 868
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 534
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2239
type: DSZ, layer: 1, pos: 2347
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2951
type: DSZ, layer: 1, pos: 3206
type: DSZ, layer: 1, pos: 2305
type: DSZ, layer: 1, pos: 2077
type: DSZ, layer: 1, pos: 2205
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 692
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2235
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2063
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2200
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2041
type: DSZ, layer: 1, pos: 821
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 99
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 3459
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2650
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2027
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2893
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 691
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 779
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2237
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2879
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 547
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2076
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2098
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2060
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2441
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2295
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2424
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2653
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3292
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 750
type: DSZ, layer: 1, pos: 707
type: DSZ, layer: 1, pos: 3114
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2221
type: DSZ, layer: 1, pos: 511
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2646
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 129
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 302
type: DSZ, layer: 1, pos: 2834
type: DSZ, layer: 1, pos: 3102
type: DSZ, layer: 1, pos: 2061
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 106
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 3113
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2145
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 723
type: DSZ, layer: 1, pos: 766
type: DSZ, layer: 1, pos: 2156
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 533
type: DSZ, layer: 1, pos: 861
type: DSZ, layer: 1, pos: 2114
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 3472
type: DSZ, layer: 1, pos: 215

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2035

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0145939, upper bound: 0.0146132
time: 4.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0145778, upper bound: 0.0146292
time: 4.19 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 14.73 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.73
Output dim: 0, lower bound: -0.0146327, upper bound: 0.0146230
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.73
Output dim: 0, lower bound: -0.0146296, upper bound: 0.0146260
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.73
Output dim: 0, lower bound: -0.0146296, upper bound: 0.0146222
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.73
Output dim: 0, lower bound: -0.0146295, upper bound: 0.0146222
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.73
Output dim: 0, lower bound: -0.0146352, upper bound: 0.0146282
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.73
Output dim: 0, lower bound: -0.0146352, upper bound: 0.0146282
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.73
Output dim: 0, lower bound: -0.0146317, upper bound: 0.0146246
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.73
Output dim: 0, lower bound: -0.0146316, upper bound: 0.0146248
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.73
Output dim: 0, lower bound: -0.0146324, upper bound: 0.0144942
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 14.73
Output dim: 0, lower bound: -0.0145339, upper bound: 0.0145927
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 14.73
Output dim: 0, lower bound: -0.0146290, upper bound: 0.0145847
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.73
Output dim: 0, lower bound: -0.0146270, upper bound: 0.0145867
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 14.73
Output dim: 0, lower bound: -0.0145939, upper bound: 0.0146132
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 14.73
Output dim: 0, lower bound: -0.0145778, upper bound: 0.0146292
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -0.0146284, upper bound: 0.0146151
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -0.0146275, upper bound: 0.0146161
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -0.0146299, upper bound: 0.0146029
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -0.0145953, upper bound: 0.0146357
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -0.0146097, upper bound: 0.0146361
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -0.0146097, upper bound: 0.0146362
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -0.0146358, upper bound: 0.0146170
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -0.0146358, upper bound: 0.0146168
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -0.0146323, upper bound: 0.0146223
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -0.0146323, upper bound: 0.0146222
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -0.0146344, upper bound: 0.0146125
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -0.0146348, upper bound: 0.0146121
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -0.0146254, upper bound: 0.0146116
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -0.0146041, upper bound: 0.0146279
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -0.0146062, upper bound: 0.0146277
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.73
Output dim: 0, lower bound: -0.0146044, upper bound: 0.0146294

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 32.72 + 1774.06 = 1806.78 seconds
