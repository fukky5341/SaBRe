## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 13)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.005478520999999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1617962, 0.1617962)
1: (-4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1696471, 0.1696471)
2: (-2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0311683, 0.0311683)
3: (0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0093332, 0.0093332)
4: (-1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0576137, 0.0576137)
5: (0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0179568, 0.0179568)
6: (-1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622505, 0.0622504)
7: (0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0386543, 0.0386543)
8: (-3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1673808, 0.1673808)
9: (-4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1499793, 0.1499793)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.14 + 40.14 = 47.28 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0054881, upper bound: 0.0054897

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 338
type: B, layer: 1, pos: 338
type: A, layer: 1, pos: 326
type: B, layer: 1, pos: 326
type: A, layer: 1, pos: 3039
type: B, layer: 1, pos: 3039
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 2093
type: B, layer: 1, pos: 2093
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2167
type: B, layer: 1, pos: 2167
type: A, layer: 1, pos: 2597
type: B, layer: 1, pos: 2597
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 2633
type: A, layer: 1, pos: 2648
type: B, layer: 1, pos: 2648
type: A, layer: 1, pos: 2122
type: B, layer: 1, pos: 2122
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2632
type: B, layer: 1, pos: 2632
type: A, layer: 1, pos: 2539
type: B, layer: 1, pos: 2539
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 2228
type: B, layer: 1, pos: 2228
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: A, layer: 1, pos: 2647
type: B, layer: 1, pos: 2647
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: B, layer: 1, pos: 2202
type: A, layer: 1, pos: 3578
type: B, layer: 1, pos: 3578
type: A, layer: 1, pos: 2442
type: B, layer: 1, pos: 2442
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 2180
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2158
type: B, layer: 1, pos: 2158
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 3502
type: A, layer: 1, pos: 2510
type: B, layer: 1, pos: 2510
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 2291
type: B, layer: 1, pos: 2291
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 2676
type: B, layer: 1, pos: 2676
type: A, layer: 1, pos: 2482
type: B, layer: 1, pos: 2482
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 2113
type: B, layer: 1, pos: 2113
type: A, layer: 1, pos: 2059
type: B, layer: 1, pos: 2059
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 2182
type: B, layer: 1, pos: 2182
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 2932
type: B, layer: 1, pos: 2932
type: A, layer: 1, pos: 2293
type: B, layer: 1, pos: 2293
type: A, layer: 1, pos: 2292
type: B, layer: 1, pos: 2292
type: A, layer: 1, pos: 2185
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 2643
type: B, layer: 1, pos: 2643
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 2046
type: B, layer: 1, pos: 2046
type: A, layer: 1, pos: 2610
type: B, layer: 1, pos: 2610
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 2307
type: B, layer: 1, pos: 2307
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2300
type: B, layer: 1, pos: 2300
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 2181
type: B, layer: 1, pos: 2181
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 3123
type: B, layer: 1, pos: 3123
type: A, layer: 1, pos: 2263
type: B, layer: 1, pos: 2263
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 2116
type: B, layer: 1, pos: 2116
type: A, layer: 1, pos: 2257
type: B, layer: 1, pos: 2257
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 2262
type: B, layer: 1, pos: 2262
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 905
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 932
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2984
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2240
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2984

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 338

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054880, upper bound: 0.0054605
time: 32.01 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054879, upper bound: 0.0054886
time: 2.96 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 35.06 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 35.06
Output dim: 3, lower bound: -0.0054880, upper bound: 0.0054605
NS_A2, status: Status.UNKNOWN, split count: 1, time: 35.06
Output dim: 3, lower bound: -0.0054879, upper bound: 0.0054886

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -6.0887880, -5.3182716, -6.0888052, -5.3182683, -0.1614555, 0.1614925
1: -4.1810040, -3.4585729, -4.1810136, -3.4585705, -0.1694979, 0.1695094
2: -2.4613295, -2.2315247, -2.4613686, -2.2315245, -0.0309860, 0.0310122
3: 0.0919736, 0.1945722, 0.0919725, 0.1946134, -0.0091138, 0.0090843
4: -1.3954426, -1.0505964, -1.3956528, -1.0505964, -0.0564725, 0.0566831
5: 0.4018854, 0.5859153, 0.4018853, 0.5859774, -0.0176913, 0.0176517
6: -1.2436389, -1.0264900, -1.2439283, -1.0264900, -0.0597812, 0.0601742
7: 0.1242563, 0.4745712, 0.1242563, 0.4747986, -0.0373058, 0.0370235
8: -3.7376776, -2.8694553, -3.7377088, -2.8694549, -0.1664406, 0.1665112
9: -4.5816536, -3.9145629, -4.5816545, -3.9145620, -0.1498511, 0.1498666

Time for backsubstitution: 5.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 326
type: A, layer: 1, pos: 326
type: B, layer: 1, pos: 3039
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2093
type: B, layer: 1, pos: 2093
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2167
type: A, layer: 1, pos: 2167
type: B, layer: 1, pos: 2597
type: A, layer: 1, pos: 2597
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 2648
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2122
type: B, layer: 1, pos: 2122
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2632
type: B, layer: 1, pos: 2632
type: A, layer: 1, pos: 2539
type: B, layer: 1, pos: 2539
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 2228
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: A, layer: 1, pos: 2647
type: B, layer: 1, pos: 2647
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 3578
type: A, layer: 1, pos: 3578
type: B, layer: 1, pos: 2442
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 2180
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2158
type: A, layer: 1, pos: 2158
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 3502
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 2510
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2291
type: A, layer: 1, pos: 2291
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2676
type: B, layer: 1, pos: 2676
type: A, layer: 1, pos: 2482
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 2113
type: A, layer: 1, pos: 2113
type: B, layer: 1, pos: 2059
type: A, layer: 1, pos: 2059
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 2182
type: A, layer: 1, pos: 2182
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 2932
type: A, layer: 1, pos: 2932
type: B, layer: 1, pos: 2293
type: A, layer: 1, pos: 2293
type: B, layer: 1, pos: 2292
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 2643
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2046
type: B, layer: 1, pos: 2046
type: A, layer: 1, pos: 2610
type: B, layer: 1, pos: 2610
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2307
type: B, layer: 1, pos: 2307
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2300
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 2181
type: A, layer: 1, pos: 2181
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 3123
type: A, layer: 1, pos: 3123
type: B, layer: 1, pos: 2263
type: A, layer: 1, pos: 2263
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 2116
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 2257
type: A, layer: 1, pos: 2257
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 2262
type: A, layer: 1, pos: 2262
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 905
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2984
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2240
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2984

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 326

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054596, upper bound: 0.0054600
time: 3.38 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054858, upper bound: 0.0054570
time: 63.42 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -6.0884843, -5.3183556, -6.0885463, -5.3182554, -0.1614591, 0.1619604
1: -4.1807041, -3.4587784, -4.1807675, -3.4585626, -0.1694873, 0.1695666
2: -2.4615226, -2.2312853, -2.4615252, -2.2315228, -0.0310114, 0.0313903
3: 0.0916794, 0.1948039, 0.0919674, 0.1948042, -0.0096256, 0.0091186
4: -1.3965762, -1.0492629, -1.3965731, -1.0505964, -0.0566340, 0.0589283
5: 0.4014791, 0.5862483, 0.4018839, 0.5862487, -0.0183330, 0.0176901
6: -1.2455332, -1.0262134, -1.2454050, -1.0264900, -0.0617767, 0.0623560
7: 0.1231095, 0.4753745, 0.1242563, 0.4754355, -0.0406001, 0.0371309
8: -3.7371154, -2.8697238, -3.7372589, -2.8694506, -0.1664090, 0.1683277
9: -4.5814328, -3.9147310, -4.5814743, -3.9145594, -0.1498394, 0.1500033

Time for backsubstitution: 5.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 326
type: A, layer: 1, pos: 326
type: B, layer: 1, pos: 3039
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2093
type: B, layer: 1, pos: 2093
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2167
type: A, layer: 1, pos: 2167
type: B, layer: 1, pos: 2597
type: A, layer: 1, pos: 2597
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 2648
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2122
type: B, layer: 1, pos: 2122
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 2632
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2539
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 2228
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: A, layer: 1, pos: 2647
type: B, layer: 1, pos: 2647
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 3578
type: A, layer: 1, pos: 3578
type: B, layer: 1, pos: 2442
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 2180
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2158
type: A, layer: 1, pos: 2158
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 3502
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 2510
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2291
type: A, layer: 1, pos: 2291
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2676
type: B, layer: 1, pos: 2676
type: A, layer: 1, pos: 2482
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 2113
type: A, layer: 1, pos: 2113
type: B, layer: 1, pos: 2059
type: A, layer: 1, pos: 2059
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 2182
type: A, layer: 1, pos: 2182
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 2932
type: A, layer: 1, pos: 2932
type: B, layer: 1, pos: 2293
type: A, layer: 1, pos: 2293
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 2292
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 2643
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2046
type: B, layer: 1, pos: 2046
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2307
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2300
type: B, layer: 1, pos: 2307
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2181
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 2181
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 3123
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 3123
type: B, layer: 1, pos: 2263
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2116
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2257
type: A, layer: 1, pos: 2262
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: B, layer: 1, pos: 2116
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 905
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2984
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2240
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2984
type: A, layer: 1, pos: 932

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 326

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054595, upper bound: 0.0054874
time: 3.47 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054857, upper bound: 0.0054875
time: 4.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 13.34 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 13.34
Output dim: 3, lower bound: -0.0054596, upper bound: 0.0054600
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 13.34
Output dim: 3, lower bound: -0.0054858, upper bound: 0.0054570
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 13.34
Output dim: 3, lower bound: -0.0054595, upper bound: 0.0054874
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 13.34
Output dim: 3, lower bound: -0.0054857, upper bound: 0.0054875

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -6.0887880, -5.3182769, -6.0888052, -5.3182755, -0.1581481, 0.1614916
1: -4.1810040, -3.4585779, -4.1810136, -3.4585772, -0.1659816, 0.1695083
2: -2.4613295, -2.2315259, -2.4613686, -2.2315259, -0.0289956, 0.0310119
3: 0.0919735, 0.1945629, 0.0919725, 0.1946011, -0.0084898, 0.0090837
4: -1.3954309, -1.0505964, -1.3956374, -1.0505964, -0.0564724, 0.0529266
5: 0.4018854, 0.5859094, 0.4018852, 0.5859702, -0.0167303, 0.0176516
6: -1.2436330, -1.0264900, -1.2439212, -1.0264900, -0.0597770, 0.0586685
7: 0.1242569, 0.4745712, 0.1242570, 0.4747987, -0.0373051, 0.0334375
8: -3.7376776, -2.8694696, -3.7377088, -2.8694742, -0.1658750, 0.1665109
9: -4.5816536, -3.9145663, -4.5816545, -3.9145658, -0.1478316, 0.1498661

Time for backsubstitution: 5.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3039
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 2093
type: A, layer: 1, pos: 2093
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 326
type: A, layer: 1, pos: 2167
type: B, layer: 1, pos: 2167
type: A, layer: 1, pos: 2597
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 338
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 2633
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 2648
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2122
type: A, layer: 1, pos: 2122
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2632
type: A, layer: 1, pos: 2632
type: B, layer: 1, pos: 2539
type: A, layer: 1, pos: 2539
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 2228
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3274
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 2647
type: A, layer: 1, pos: 2647
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 3578
type: B, layer: 1, pos: 3578
type: A, layer: 1, pos: 2442
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2158
type: B, layer: 1, pos: 2158
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 3502
type: A, layer: 1, pos: 2510
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2291
type: B, layer: 1, pos: 2291
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2676
type: A, layer: 1, pos: 2676
type: B, layer: 1, pos: 2482
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 2113
type: B, layer: 1, pos: 2113
type: A, layer: 1, pos: 2059
type: B, layer: 1, pos: 2059
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 2182
type: B, layer: 1, pos: 2182
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 2932
type: B, layer: 1, pos: 2932
type: A, layer: 1, pos: 2293
type: B, layer: 1, pos: 2293
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 2185
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2292
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2643
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 2046
type: A, layer: 1, pos: 2046
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2307
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 2307
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2181
type: B, layer: 1, pos: 2181
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 3123
type: B, layer: 1, pos: 3123
type: A, layer: 1, pos: 2263
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2116
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 2257
type: A, layer: 1, pos: 2262
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 2262
type: A, layer: 1, pos: 2116
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 905
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2984
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2240
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 932

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3039

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054803, upper bound: 0.0054469
time: 2.64 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054756, upper bound: 0.0054459
time: 96.65 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -6.0884843, -5.3203278, -6.0876794, -5.3207011, -0.1589375, 0.1590208
1: -4.1807036, -3.4608331, -4.1798816, -3.4611204, -0.1668109, 0.1664465
2: -2.4615216, -2.2324805, -2.4609535, -2.2330420, -0.0294871, 0.0296167
3: 0.0916837, 0.1944515, 0.0920878, 0.1943780, -0.0091398, 0.0085615
4: -1.3943450, -1.0492629, -1.3937819, -1.0515682, -0.0532736, 0.0560355
5: 0.4014798, 0.5857182, 0.4020633, 0.5856097, -0.0176041, 0.0168424
6: -1.2435343, -1.0262134, -1.2428324, -1.0269024, -0.0588357, 0.0591696
7: 0.1252691, 0.4753697, 0.1269826, 0.4744109, -0.0374021, 0.0343785
8: -3.7371154, -2.8700206, -3.7371686, -2.8698285, -0.1659810, 0.1678291
9: -4.5814328, -3.9159389, -4.5809331, -3.9160633, -0.1482988, 0.1482075

Time for backsubstitution: 5.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3039
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2093
type: B, layer: 1, pos: 2093
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2167
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 326
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 2597
type: A, layer: 1, pos: 2597
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 2648
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2122
type: B, layer: 1, pos: 2122
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 2632
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2539
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 2228
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: A, layer: 1, pos: 2647
type: B, layer: 1, pos: 2647
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 3578
type: A, layer: 1, pos: 3578
type: B, layer: 1, pos: 2442
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 2180
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2158
type: A, layer: 1, pos: 2158
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 3502
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 2510
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2291
type: A, layer: 1, pos: 2291
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2676
type: B, layer: 1, pos: 2676
type: A, layer: 1, pos: 2482
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 2113
type: A, layer: 1, pos: 2113
type: B, layer: 1, pos: 2059
type: A, layer: 1, pos: 2059
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 2182
type: A, layer: 1, pos: 2182
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 2932
type: A, layer: 1, pos: 2932
type: B, layer: 1, pos: 2293
type: A, layer: 1, pos: 2293
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 2292
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 2643
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2046
type: B, layer: 1, pos: 2046
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2307
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2300
type: B, layer: 1, pos: 2307
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2181
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 2181
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 3123
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 3123
type: B, layer: 1, pos: 2263
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2116
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2257
type: A, layer: 1, pos: 2262
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 905
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 2116
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2984
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2240
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2984
type: A, layer: 1, pos: 932

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 3039

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054474, upper bound: 0.0054822
time: 4.68 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054475, upper bound: 0.0054759
time: 7.64 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -6.0884843, -5.3183608, -6.0885463, -5.3182626, -0.1581518, 0.1619595
1: -4.1807041, -3.4587839, -4.1807675, -3.4585695, -0.1659711, 0.1695656
2: -2.4615226, -2.2312868, -2.4615252, -2.2315245, -0.0290211, 0.0313900
3: 0.0916795, 0.1947946, 0.0919674, 0.1947919, -0.0090019, 0.0091180
4: -1.3965645, -1.0492629, -1.3965578, -1.0505964, -0.0566339, 0.0551719
5: 0.4014791, 0.5862426, 0.4018839, 0.5862412, -0.0173720, 0.0176900
6: -1.2455275, -1.0262134, -1.2453973, -1.0264900, -0.0617725, 0.0608503
7: 0.1231101, 0.4753746, 0.1242570, 0.4754355, -0.0405994, 0.0335454
8: -3.7371154, -2.8697386, -3.7372589, -2.8694699, -0.1658434, 0.1683273
9: -4.5814328, -3.9147334, -4.5814743, -3.9145627, -0.1478198, 0.1500028

Time for backsubstitution: 5.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3039
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2093
type: B, layer: 1, pos: 2093
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2167
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 326
type: B, layer: 1, pos: 2597
type: A, layer: 1, pos: 2597
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 2648
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2122
type: B, layer: 1, pos: 2122
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2632
type: B, layer: 1, pos: 2632
type: A, layer: 1, pos: 2539
type: B, layer: 1, pos: 2539
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 2228
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: A, layer: 1, pos: 2647
type: B, layer: 1, pos: 2647
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 3578
type: A, layer: 1, pos: 3578
type: B, layer: 1, pos: 2442
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 2180
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2158
type: A, layer: 1, pos: 2158
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 3502
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 2510
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2291
type: A, layer: 1, pos: 2291
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2676
type: B, layer: 1, pos: 2676
type: A, layer: 1, pos: 2482
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 2113
type: A, layer: 1, pos: 2113
type: B, layer: 1, pos: 2059
type: A, layer: 1, pos: 2059
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 2182
type: A, layer: 1, pos: 2182
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 2932
type: A, layer: 1, pos: 2932
type: B, layer: 1, pos: 2293
type: A, layer: 1, pos: 2293
type: B, layer: 1, pos: 2292
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2185
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 2643
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2046
type: B, layer: 1, pos: 2046
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 2307
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 2307
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2181
type: A, layer: 1, pos: 2181
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 3123
type: A, layer: 1, pos: 3123
type: B, layer: 1, pos: 2263
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2116
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2257
type: A, layer: 1, pos: 2262
type: B, layer: 1, pos: 2116
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 905
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2984
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2240
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2984
type: A, layer: 1, pos: 932

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 3039

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054743, upper bound: 0.0054797
time: 4.38 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054744, upper bound: 0.0054759
time: 11.88 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 21.87 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.87
Output dim: 3, lower bound: -0.0054803, upper bound: 0.0054469
NS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 21.87
Output dim: 3, lower bound: -0.0054756, upper bound: 0.0054459
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 21.87
Output dim: 3, lower bound: -0.0054474, upper bound: 0.0054822
NS_A2_B1_B2, status: Status.VERIFIED, split count: 3, time: 21.87
Output dim: 3, lower bound: -0.0054475, upper bound: 0.0054759
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 21.87
Output dim: 3, lower bound: -0.0054743, upper bound: 0.0054797
NS_A2_B2_B2, status: Status.VERIFIED, split count: 3, time: 21.87
Output dim: 3, lower bound: -0.0054744, upper bound: 0.0054759

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.0887880, -5.3194461, -6.0888052, -5.3193488, -0.1552174, 0.1579093
1: -4.1810045, -3.4596076, -4.1810136, -3.4595220, -0.1636515, 0.1666586
2: -2.4613292, -2.2315974, -2.4613686, -2.2315888, -0.0288139, 0.0307929
3: 0.0920154, 0.1945629, 0.0920096, 0.1946011, -0.0083414, 0.0089627
4: -1.3953745, -1.0505964, -1.3955883, -1.0505964, -0.0563904, 0.0528538
5: 0.4019476, 0.5859094, 0.4019398, 0.5859702, -0.0165465, 0.0175015
6: -1.2431145, -1.0264900, -1.2434721, -1.0264900, -0.0589565, 0.0579266
7: 0.1242569, 0.4745482, 0.1242570, 0.4747790, -0.0372348, 0.0333515
8: -3.7376776, -2.8698599, -3.7377088, -2.8698220, -0.1645228, 0.1648678
9: -4.5816536, -3.9150820, -4.5816545, -3.9150131, -0.1468360, 0.1486519

Time for backsubstitution: 5.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2348
type: A, layer: 1, pos: 2348
type: A, layer: 1, pos: 769
type: B, layer: 1, pos: 769
type: B, layer: 1, pos: 2093
type: A, layer: 1, pos: 2093
type: B, layer: 1, pos: 2612
type: A, layer: 1, pos: 2612
type: A, layer: 1, pos: 326
type: A, layer: 1, pos: 2167
type: B, layer: 1, pos: 2167
type: A, layer: 1, pos: 2597
type: B, layer: 1, pos: 2597
type: B, layer: 1, pos: 338
type: A, layer: 1, pos: 767
type: B, layer: 1, pos: 767
type: B, layer: 1, pos: 2633
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 2648
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2122
type: A, layer: 1, pos: 2122
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2632
type: A, layer: 1, pos: 2632
type: B, layer: 1, pos: 2539
type: A, layer: 1, pos: 2539
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 2228
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3274
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 2647
type: A, layer: 1, pos: 2647
type: B, layer: 1, pos: 488
type: A, layer: 1, pos: 488
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 205
type: B, layer: 1, pos: 205
type: B, layer: 1, pos: 2202
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 3578
type: B, layer: 1, pos: 3578
type: A, layer: 1, pos: 2442
type: B, layer: 1, pos: 2442
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2180
type: B, layer: 1, pos: 2196
type: A, layer: 1, pos: 2196
type: A, layer: 1, pos: 2158
type: B, layer: 1, pos: 2158
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 3502
type: A, layer: 1, pos: 2510
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 2497
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2291
type: B, layer: 1, pos: 2291
type: A, layer: 1, pos: 878
type: B, layer: 1, pos: 878
type: B, layer: 1, pos: 2676
type: A, layer: 1, pos: 2676
type: B, layer: 1, pos: 2482
type: A, layer: 1, pos: 2482
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 2113
type: B, layer: 1, pos: 2113
type: A, layer: 1, pos: 2059
type: B, layer: 1, pos: 2059
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 169
type: A, layer: 1, pos: 169
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 2182
type: B, layer: 1, pos: 2182
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 2932
type: B, layer: 1, pos: 2932
type: A, layer: 1, pos: 2293
type: B, layer: 1, pos: 2293
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 2185
type: B, layer: 1, pos: 2292
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 165
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2643
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 2643
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 2046
type: A, layer: 1, pos: 2046
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 2307
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 2307
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 2181
type: B, layer: 1, pos: 2181
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 3123
type: B, layer: 1, pos: 3123
type: A, layer: 1, pos: 2263
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 727
type: B, layer: 1, pos: 727
type: B, layer: 1, pos: 2116
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 2257
type: A, layer: 1, pos: 2262
type: A, layer: 1, pos: 883
type: B, layer: 1, pos: 883
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 2262
type: A, layer: 1, pos: 2116
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 905
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 837
type: A, layer: 1, pos: 837
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 976
type: B, layer: 1, pos: 976
type: B, layer: 1, pos: 903
type: A, layer: 1, pos: 903
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2984
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2240
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2984
type: B, layer: 1, pos: 932

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 2164

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054725, upper bound: 0.0054443
time: 2.62 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054727, upper bound: 0.0054374
time: 26.75 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -6.0884843, -5.3213992, -6.0876794, -5.3218713, -0.1553554, 0.1561061
1: -4.1807032, -3.4617782, -4.1798811, -3.4621501, -0.1639614, 0.1641298
2: -2.4615216, -2.2325435, -2.4609532, -2.2331135, -0.0292682, 0.0294364
3: 0.0917206, 0.1944515, 0.0921296, 0.1943780, -0.0090201, 0.0084131
4: -1.3942950, -1.0492629, -1.3937249, -1.0515682, -0.0532007, 0.0559558
5: 0.4015353, 0.5857182, 0.4021255, 0.5856097, -0.0174549, 0.0166587
6: -1.2430896, -1.0262134, -1.2423172, -1.0269024, -0.0581004, 0.0583575
7: 0.1252691, 0.4753490, 0.1269826, 0.4743867, -0.0373291, 0.0343081
8: -3.7371154, -2.8703687, -3.7371686, -2.8702178, -0.1643380, 0.1664876
9: -4.5814328, -3.9163864, -4.5809331, -3.9165802, -0.1470847, 0.1472128

Time for backsubstitution: 5.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2093
type: B, layer: 1, pos: 2093
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2167
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 326
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 2597
type: A, layer: 1, pos: 2597
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 2648
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2122
type: B, layer: 1, pos: 2122
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2539
type: A, layer: 1, pos: 2632
type: B, layer: 1, pos: 2632
type: B, layer: 1, pos: 2539
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 2228
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: A, layer: 1, pos: 2647
type: B, layer: 1, pos: 2647
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 3578
type: A, layer: 1, pos: 3578
type: B, layer: 1, pos: 2442
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 2180
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2158
type: A, layer: 1, pos: 2158
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 3502
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 2510
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2291
type: A, layer: 1, pos: 2291
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2676
type: B, layer: 1, pos: 2676
type: A, layer: 1, pos: 2482
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 2113
type: A, layer: 1, pos: 2113
type: B, layer: 1, pos: 2059
type: A, layer: 1, pos: 2059
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 2182
type: A, layer: 1, pos: 2182
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 2932
type: A, layer: 1, pos: 2932
type: B, layer: 1, pos: 2293
type: A, layer: 1, pos: 2293
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 2292
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 2643
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2046
type: B, layer: 1, pos: 2046
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 695
type: A, layer: 1, pos: 2307
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2300
type: B, layer: 1, pos: 2307
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2181
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 2181
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 57
type: B, layer: 1, pos: 3123
type: A, layer: 1, pos: 57
type: A, layer: 1, pos: 3123
type: B, layer: 1, pos: 2263
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2116
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2257
type: A, layer: 1, pos: 2262
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 905
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 933
type: B, layer: 1, pos: 2116
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2984
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2240
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2984
type: A, layer: 1, pos: 932

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2164

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054443, upper bound: 0.0054736
time: 2.86 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054399, upper bound: 0.0054745
time: 5.06 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -6.0884843, -5.3194337, -6.0885463, -5.3194327, -0.1545697, 0.1590447
1: -4.1807041, -3.4597292, -4.1807675, -3.4595988, -0.1631217, 0.1672488
2: -2.4615223, -2.2313497, -2.4615252, -2.2315962, -0.0288021, 0.0312098
3: 0.0917163, 0.1947946, 0.0920093, 0.1947919, -0.0088822, 0.0089697
4: -1.3965144, -1.0492629, -1.3965003, -1.0505964, -0.0565610, 0.0550923
5: 0.4015346, 0.5862426, 0.4019461, 0.5862412, -0.0172228, 0.0175062
6: -1.2450812, -1.0262134, -1.2448804, -1.0264900, -0.0610381, 0.0600404
7: 0.1231101, 0.4753536, 0.1242570, 0.4754113, -0.0405265, 0.0334750
8: -3.7371154, -2.8700867, -3.7372589, -2.8698597, -0.1642004, 0.1669858
9: -4.5814328, -3.9151819, -4.5814743, -3.9150791, -0.1466056, 0.1490081

Time for backsubstitution: 5.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2164
type: B, layer: 1, pos: 2164
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2348
type: B, layer: 1, pos: 2348
type: B, layer: 1, pos: 769
type: A, layer: 1, pos: 769
type: A, layer: 1, pos: 2093
type: B, layer: 1, pos: 2093
type: A, layer: 1, pos: 2612
type: B, layer: 1, pos: 2612
type: B, layer: 1, pos: 2167
type: A, layer: 1, pos: 2167
type: A, layer: 1, pos: 326
type: B, layer: 1, pos: 2597
type: A, layer: 1, pos: 2597
type: B, layer: 1, pos: 338
type: B, layer: 1, pos: 767
type: A, layer: 1, pos: 767
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 2648
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2122
type: B, layer: 1, pos: 2122
type: A, layer: 1, pos: 2557
type: B, layer: 1, pos: 2557
type: A, layer: 1, pos: 2632
type: B, layer: 1, pos: 2632
type: A, layer: 1, pos: 2539
type: B, layer: 1, pos: 2539
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 203
type: A, layer: 1, pos: 203
type: B, layer: 1, pos: 2228
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3274
type: B, layer: 1, pos: 3274
type: A, layer: 1, pos: 2647
type: B, layer: 1, pos: 2647
type: A, layer: 1, pos: 488
type: B, layer: 1, pos: 488
type: B, layer: 1, pos: 71
type: A, layer: 1, pos: 71
type: B, layer: 1, pos: 205
type: A, layer: 1, pos: 205
type: A, layer: 1, pos: 2202
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 3578
type: A, layer: 1, pos: 3578
type: B, layer: 1, pos: 2442
type: A, layer: 1, pos: 2442
type: A, layer: 1, pos: 548
type: B, layer: 1, pos: 548
type: A, layer: 1, pos: 2180
type: B, layer: 1, pos: 2180
type: A, layer: 1, pos: 2196
type: B, layer: 1, pos: 2196
type: B, layer: 1, pos: 2158
type: A, layer: 1, pos: 2158
type: B, layer: 1, pos: 2996
type: A, layer: 1, pos: 2996
type: B, layer: 1, pos: 3502
type: A, layer: 1, pos: 3502
type: B, layer: 1, pos: 2510
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 2497
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2291
type: A, layer: 1, pos: 2291
type: B, layer: 1, pos: 878
type: A, layer: 1, pos: 878
type: A, layer: 1, pos: 2676
type: B, layer: 1, pos: 2676
type: A, layer: 1, pos: 2482
type: B, layer: 1, pos: 2482
type: B, layer: 1, pos: 740
type: A, layer: 1, pos: 740
type: B, layer: 1, pos: 2113
type: A, layer: 1, pos: 2113
type: B, layer: 1, pos: 2059
type: A, layer: 1, pos: 2059
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 169
type: B, layer: 1, pos: 169
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 2182
type: A, layer: 1, pos: 2182
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 2932
type: A, layer: 1, pos: 2932
type: B, layer: 1, pos: 2293
type: A, layer: 1, pos: 2293
type: B, layer: 1, pos: 2292
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 2292
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2185
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 165
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 2643
type: A, layer: 1, pos: 777
type: B, layer: 1, pos: 777
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2643
type: A, layer: 1, pos: 2046
type: B, layer: 1, pos: 2046
type: A, layer: 1, pos: 2610
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 695
type: A, layer: 1, pos: 695
type: B, layer: 1, pos: 2610
type: B, layer: 1, pos: 778
type: A, layer: 1, pos: 778
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2307
type: B, layer: 1, pos: 2307
type: A, layer: 1, pos: 2300
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 2300
type: B, layer: 1, pos: 2181
type: A, layer: 1, pos: 2181
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 702
type: A, layer: 1, pos: 702
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 3123
type: A, layer: 1, pos: 3123
type: B, layer: 1, pos: 2263
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 765
type: A, layer: 1, pos: 765
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 857
type: A, layer: 1, pos: 857
type: B, layer: 1, pos: 727
type: A, layer: 1, pos: 727
type: A, layer: 1, pos: 2116
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 2257
type: B, layer: 1, pos: 2262
type: B, layer: 1, pos: 883
type: A, layer: 1, pos: 883
type: A, layer: 1, pos: 2257
type: A, layer: 1, pos: 2262
type: B, layer: 1, pos: 2116
type: A, layer: 1, pos: 918
type: B, layer: 1, pos: 918
type: A, layer: 1, pos: 905
type: B, layer: 1, pos: 905
type: A, layer: 1, pos: 837
type: B, layer: 1, pos: 837
type: B, layer: 1, pos: 948
type: A, layer: 1, pos: 948
type: B, layer: 1, pos: 977
type: A, layer: 1, pos: 977
type: B, layer: 1, pos: 962
type: A, layer: 1, pos: 962
type: B, layer: 1, pos: 947
type: A, layer: 1, pos: 947
type: B, layer: 1, pos: 963
type: A, layer: 1, pos: 963
type: B, layer: 1, pos: 933
type: A, layer: 1, pos: 933
type: B, layer: 1, pos: 976
type: A, layer: 1, pos: 976
type: A, layer: 1, pos: 903
type: B, layer: 1, pos: 903
type: B, layer: 1, pos: 932
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 223
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 884
type: A, layer: 1, pos: 892
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 898
type: A, layer: 1, pos: 989
type: A, layer: 1, pos: 2039
type: A, layer: 1, pos: 2069
type: A, layer: 1, pos: 2129
type: A, layer: 1, pos: 2240
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 2984
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 223
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 884
type: B, layer: 1, pos: 892
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 898
type: B, layer: 1, pos: 989
type: B, layer: 1, pos: 2039
type: B, layer: 1, pos: 2069
type: B, layer: 1, pos: 2129
type: B, layer: 1, pos: 2240
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 2984
type: A, layer: 1, pos: 932

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 2164

## Relational analysis of NS_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054709, upper bound: 0.0054729
time: 57.72 seconds

## Relational analysis of NS_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054669, upper bound: 0.0054739
time: 7.64 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 70.94 seconds
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 70.94
Output dim: 3, lower bound: -0.0054725, upper bound: 0.0054443
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 70.94
Output dim: 3, lower bound: -0.0054727, upper bound: 0.0054374
NS_A2_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 70.94
Output dim: 3, lower bound: -0.0054443, upper bound: 0.0054736
NS_A2_B1_B1_A2, status: Status.VERIFIED, split count: 4, time: 70.94
Output dim: 3, lower bound: -0.0054399, upper bound: 0.0054745
NS_A2_B2_B1_A1, status: Status.VERIFIED, split count: 4, time: 70.94
Output dim: 3, lower bound: -0.0054709, upper bound: 0.0054729
NS_A2_B2_B1_A2, status: Status.VERIFIED, split count: 4, time: 70.94
Output dim: 3, lower bound: -0.0054669, upper bound: 0.0054739

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 47.28 + 384.93 = 432.21 seconds
