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
execution time: IAR + RelationalAnalysis = 8.35 + 39.00 = 47.34 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0054881, upper bound: 0.0054897

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2632

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2348

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054812, upper bound: 0.0054863
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054844, upper bound: 0.0054812
time: 110.38 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 113.34 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 113.34
Output dim: 3, lower bound: -0.0054812, upper bound: 0.0054863
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 113.34
Output dim: 3, lower bound: -0.0054844, upper bound: 0.0054812

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1567394, 0.1566042
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1654142, 0.1653483
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309441, 0.0309341
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091359, 0.0091374
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0575276, 0.0575279
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177249, 0.0177274
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612931, 0.0613351
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0386239, 0.0386227
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1648840, 0.1649192
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1482150, 0.1481876

Time for backsubstitution: 6.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2228

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2158

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054783, upper bound: 0.0054870
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054811, upper bound: 0.0054813
time: 8.78 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1566042, 0.1567394
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1653483, 0.1654142
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309341, 0.0309441
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091374, 0.0091359
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0575279, 0.0575276
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177274, 0.0177249
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0613351, 0.0612931
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0386227, 0.0386239
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1649192, 0.1648840
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1481877, 0.1482150

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 203

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2262

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054845, upper bound: 0.0054826
time: 16.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054850, upper bound: 0.0054826
time: 15.03 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 37.86 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 37.86
Output dim: 3, lower bound: -0.0054783, upper bound: 0.0054870
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 37.86
Output dim: 3, lower bound: -0.0054811, upper bound: 0.0054813
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 37.86
Output dim: 3, lower bound: -0.0054845, upper bound: 0.0054826
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 37.86
Output dim: 3, lower bound: -0.0054850, upper bound: 0.0054826

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1564527, 0.1562591
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1651005, 0.1649777
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309332, 0.0309210
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091332, 0.0091356
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0575220, 0.0575221
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177151, 0.0177208
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612910, 0.0613329
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385996, 0.0386018
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1644811, 0.1644107
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1478903, 0.1477810

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 169

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054781, upper bound: 0.0054800
time: 19.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054749, upper bound: 0.0054846
time: 75.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1563943, 0.1563175
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1650436, 0.1650347
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309311, 0.0309231
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091342, 0.0091346
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0575217, 0.0575224
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177182, 0.0177176
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612909, 0.0613330
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0386030, 0.0385984
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1643755, 0.1645164
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1478083, 0.1478629

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2692

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054818, upper bound: 0.0054843
time: 2.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054818, upper bound: 0.0054839
time: 2.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1566005, 0.1567357
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1653481, 0.1654140
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309314, 0.0309425
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091374, 0.0091359
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0575278, 0.0575273
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177274, 0.0177248
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0613351, 0.0612931
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0386223, 0.0386235
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1649163, 0.1648811
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1481801, 0.1482101

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2293

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 765

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054842, upper bound: 0.0054831
time: 3.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054842, upper bound: 0.0054819
time: 14.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1566005, 0.1567357
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1653481, 0.1654140
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309324, 0.0309415
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091374, 0.0091359
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0575277, 0.0575274
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177274, 0.0177249
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0613351, 0.0612931
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0386223, 0.0386235
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1649164, 0.1648810
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1481827, 0.1482075

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 875

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054845, upper bound: 0.0054830
time: 10.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054845, upper bound: 0.0054835
time: 2.93 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 20.21 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 20.21
Output dim: 3, lower bound: -0.0054781, upper bound: 0.0054800
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 20.21
Output dim: 3, lower bound: -0.0054749, upper bound: 0.0054846
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 20.21
Output dim: 3, lower bound: -0.0054818, upper bound: 0.0054843
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 20.21
Output dim: 3, lower bound: -0.0054818, upper bound: 0.0054839
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 20.21
Output dim: 3, lower bound: -0.0054842, upper bound: 0.0054831
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 20.21
Output dim: 3, lower bound: -0.0054842, upper bound: 0.0054819
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 20.21
Output dim: 3, lower bound: -0.0054845, upper bound: 0.0054830
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 20.21
Output dim: 3, lower bound: -0.0054845, upper bound: 0.0054835

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1553937, 0.1551164
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1636385, 0.1635067
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0308531, 0.0308418
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091230, 0.0091254
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0574046, 0.0574130
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0176558, 0.0176612
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612571, 0.0612987
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0383727, 0.0383894
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1622458, 0.1620499
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1466132, 0.1464651

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3123

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054790, upper bound: 0.0054788
time: 6.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054790, upper bound: 0.0054810
time: 3.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1553100, 0.1552001
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1636296, 0.1635157
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0308539, 0.0308410
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091229, 0.0091254
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0574130, 0.0574047
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0176556, 0.0176615
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612569, 0.0612990
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0383872, 0.0383749
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1621203, 0.1621754
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1465745, 0.1465039

Time for backsubstitution: 6.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 905

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2996

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054734, upper bound: 0.0054849
time: 25.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054739, upper bound: 0.0054845
time: 9.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1563943, 0.1563175
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1650436, 0.1650347
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309311, 0.0309231
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091342, 0.0091346
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0575217, 0.0575224
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177182, 0.0177176
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612909, 0.0613330
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0386030, 0.0385984
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1643755, 0.1645164
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1478083, 0.1478629

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 216

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 825

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054810, upper bound: 0.0054821
time: 20.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054804, upper bound: 0.0054835
time: 5.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1563943, 0.1563175
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1650436, 0.1650347
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309311, 0.0309231
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091342, 0.0091346
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0575217, 0.0575224
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177182, 0.0177176
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612909, 0.0613330
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0386030, 0.0385984
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1643755, 0.1645164
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1478083, 0.1478629

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 702

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 778

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054810, upper bound: 0.0054819
time: 23.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054809, upper bound: 0.0054832
time: 5.92 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1562566, 0.1563732
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1651705, 0.1652266
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0308450, 0.0308602
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091368, 0.0091354
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0574798, 0.0574768
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177142, 0.0177123
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612933, 0.0612533
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0386121, 0.0386138
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1641059, 0.1640272
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1480389, 0.1480606

Time for backsubstitution: 6.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2612

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2307

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054837, upper bound: 0.0054817
time: 4.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054842, upper bound: 0.0054809
time: 19.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1562379, 0.1563919
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1651607, 0.1652364
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0308492, 0.0308560
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091368, 0.0091353
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0574772, 0.0574794
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177148, 0.0177117
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612953, 0.0612513
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0386126, 0.0386133
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1640623, 0.1640708
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1480306, 0.1480688

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 932

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2182

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054833, upper bound: 0.0054793
time: 5.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054830, upper bound: 0.0054814
time: 24.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1566001, 0.1567353
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1653478, 0.1654137
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309318, 0.0309409
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091373, 0.0091358
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0575277, 0.0575274
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177273, 0.0177248
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0613349, 0.0612929
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0386218, 0.0386230
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1649148, 0.1648794
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1481826, 0.1482074

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2539

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 918

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054843, upper bound: 0.0054813
time: 67.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054849, upper bound: 0.0054794
time: 32.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1566001, 0.1567353
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1653478, 0.1654137
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309318, 0.0309409
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091373, 0.0091358
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0575277, 0.0575274
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177273, 0.0177248
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0613349, 0.0612929
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0386218, 0.0386230
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1649148, 0.1648794
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1481826, 0.1482074

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2307

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 686

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054851, upper bound: 0.0054837
time: 4.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054848, upper bound: 0.0054817
time: 12.38 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 23.50 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 23.50
Output dim: 3, lower bound: -0.0054790, upper bound: 0.0054788
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 23.50
Output dim: 3, lower bound: -0.0054790, upper bound: 0.0054810
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 23.50
Output dim: 3, lower bound: -0.0054734, upper bound: 0.0054849
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 23.50
Output dim: 3, lower bound: -0.0054739, upper bound: 0.0054845
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 23.50
Output dim: 3, lower bound: -0.0054810, upper bound: 0.0054821
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 23.50
Output dim: 3, lower bound: -0.0054804, upper bound: 0.0054835
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 23.50
Output dim: 3, lower bound: -0.0054810, upper bound: 0.0054819
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 23.50
Output dim: 3, lower bound: -0.0054809, upper bound: 0.0054832
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 23.50
Output dim: 3, lower bound: -0.0054837, upper bound: 0.0054817
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 23.50
Output dim: 3, lower bound: -0.0054842, upper bound: 0.0054809
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 23.50
Output dim: 3, lower bound: -0.0054833, upper bound: 0.0054793
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 23.50
Output dim: 3, lower bound: -0.0054830, upper bound: 0.0054814
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 23.50
Output dim: 3, lower bound: -0.0054843, upper bound: 0.0054813
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 23.50
Output dim: 3, lower bound: -0.0054849, upper bound: 0.0054794
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 23.50
Output dim: 3, lower bound: -0.0054851, upper bound: 0.0054837
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 23.50
Output dim: 3, lower bound: -0.0054848, upper bound: 0.0054817

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1553936, 0.1551163
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1636381, 0.1635064
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0308531, 0.0308418
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091227, 0.0091251
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0574044, 0.0574128
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0176556, 0.0176609
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612571, 0.0612987
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0383727, 0.0383894
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1622456, 0.1620497
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1466130, 0.1464649

Time for backsubstitution: 6.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 208

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2692

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054785, upper bound: 0.0054787
time: 4.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054785, upper bound: 0.0054811
time: 3.51 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1553936, 0.1551162
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1636381, 0.1635064
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0308531, 0.0308418
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091227, 0.0091251
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0574044, 0.0574128
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0176555, 0.0176610
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612571, 0.0612987
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0383727, 0.0383894
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1622456, 0.1620497
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1466130, 0.1464649

Time for backsubstitution: 6.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 710

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054772, upper bound: 0.0054808
time: 2.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054772, upper bound: 0.0054822
time: 2.93 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1548934, 0.1547192
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1632772, 0.1631124
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0308450, 0.0308315
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091204, 0.0091231
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573223, 0.0573085
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0176513, 0.0176574
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612525, 0.0612957
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0383735, 0.0383599
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1618779, 0.1619164
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1465613, 0.1464889

Time for backsubstitution: 6.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 208

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054712, upper bound: 0.0054854
time: 4.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054726, upper bound: 0.0054826
time: 42.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1548290, 0.1547835
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1632263, 0.1631633
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0308444, 0.0308321
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091205, 0.0091229
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573168, 0.0573140
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0176514, 0.0176573
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612536, 0.0612947
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0383722, 0.0383612
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1618612, 0.1619331
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1465595, 0.1464907

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2039

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054738, upper bound: 0.0054837
time: 16.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054738, upper bound: 0.0054837
time: 28.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1560133, 0.1559155
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1647156, 0.1646951
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309220, 0.0309139
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091297, 0.0091301
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0575217, 0.0575224
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177008, 0.0177001
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612868, 0.0613291
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385649, 0.0385605
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1635453, 0.1636396
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1474152, 0.1474476

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 977

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 778

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054814, upper bound: 0.0054817
time: 37.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054808, upper bound: 0.0054806
time: 21.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1559923, 0.1559365
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1647040, 0.1647067
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309219, 0.0309140
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091297, 0.0091302
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0575217, 0.0575224
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177008, 0.0177002
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612869, 0.0613290
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385650, 0.0385603
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1634987, 0.1636862
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1473930, 0.1474698

Time for backsubstitution: 6.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 808

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 83

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054398, upper bound: 0.0054410
time: 98.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054398, upper bound: 0.0054409
time: 70.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1557053, 0.1556217
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1644521, 0.1644364
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0308978, 0.0308897
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091279, 0.0091282
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0575214, 0.0575221
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177022, 0.0177018
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612881, 0.0613302
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385952, 0.0385906
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1635649, 0.1636966
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1473316, 0.1473806

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 878

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2113

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054806, upper bound: 0.0054798
time: 39.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054806, upper bound: 0.0054830
time: 2.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1556985, 0.1556284
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1644453, 0.1644432
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0308976, 0.0308898
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091277, 0.0091283
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0575214, 0.0575221
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177025, 0.0177016
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612880, 0.0613302
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385952, 0.0385906
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1635556, 0.1637058
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1473260, 0.1473862

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2185

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 769

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054773, upper bound: 0.0054814
time: 4.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054798, upper bound: 0.0054779
time: 40.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1554417, 0.1555458
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1648199, 0.1648787
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0306775, 0.0306933
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091305, 0.0091293
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0572789, 0.0572775
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0176875, 0.0176861
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612810, 0.0612412
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385728, 0.0385733
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1634902, 0.1634030
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1474589, 0.1474909

Time for backsubstitution: 6.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 827

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3039

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054731, upper bound: 0.0054708
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054731, upper bound: 0.0054696
time: 9.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1554293, 0.1555583
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1648227, 0.1648760
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0306781, 0.0306928
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091307, 0.0091290
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0572805, 0.0572759
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0176881, 0.0176855
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612812, 0.0612410
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385716, 0.0385745
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1634818, 0.1634114
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1474692, 0.1474807

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 59

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054846, upper bound: 0.0054802
time: 24.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054846, upper bound: 0.0054826
time: 3.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1562314, 0.1563510
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1651452, 0.1651781
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0308440, 0.0308510
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091363, 0.0091348
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0574382, 0.0575111
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177124, 0.0177092
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612947, 0.0612507
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0386097, 0.0386106
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1641164, 0.1640045
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1480028, 0.1480003

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2932

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2185

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054821, upper bound: 0.0054799
time: 3.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054825, upper bound: 0.0054793
time: 3.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1561971, 0.1563919
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1651024, 0.1652364
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0308442, 0.0308560
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091368, 0.0091348
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0574772, 0.0574403
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177148, 0.0177093
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612953, 0.0612507
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0386126, 0.0386105
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1639960, 0.1640708
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1479621, 0.1480688

Time for backsubstitution: 6.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2164

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2122

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054793, upper bound: 0.0054799
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054808, upper bound: 0.0054766
time: 36.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1566000, 0.1567352
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1653477, 0.1654136
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309318, 0.0309408
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091373, 0.0091358
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0575277, 0.0575274
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177272, 0.0177247
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0613349, 0.0612929
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0386218, 0.0386230
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1649145, 0.1648791
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1481825, 0.1482072

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 839

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054649, upper bound: 0.0054823
time: 22.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054840, upper bound: 0.0054619
time: 7.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1566000, 0.1567352
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1653477, 0.1654136
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309318, 0.0309408
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091373, 0.0091358
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0575277, 0.0575274
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177272, 0.0177247
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0613349, 0.0612929
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0386218, 0.0386230
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1649145, 0.1648792
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1481825, 0.1482072

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 837

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 8

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054832, upper bound: 0.0054825
time: 9.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054844, upper bound: 0.0054793
time: 10.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1565508, 0.1566841
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1653310, 0.1653964
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309028, 0.0309127
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091361, 0.0091346
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0575222, 0.0575218
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177205, 0.0177181
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0613320, 0.0612900
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0386178, 0.0386189
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1648490, 0.1648112
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1481337, 0.1481587

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2039

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054836, upper bound: 0.0054809
time: 5.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054838, upper bound: 0.0054830
time: 6.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1565490, 0.1566859
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1653305, 0.1653968
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309037, 0.0309119
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091361, 0.0091346
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0575222, 0.0575218
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177206, 0.0177180
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0613320, 0.0612900
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0386177, 0.0386190
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1648467, 0.1648136
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1481345, 0.1481585

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 216

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054816, upper bound: 0.0054795
time: 5.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054816, upper bound: 0.0054831
time: 5.69 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 17.39 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054785, upper bound: 0.0054787
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054785, upper bound: 0.0054811
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054772, upper bound: 0.0054808
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054772, upper bound: 0.0054822
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054712, upper bound: 0.0054854
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054726, upper bound: 0.0054826
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054738, upper bound: 0.0054837
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054738, upper bound: 0.0054837
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054814, upper bound: 0.0054817
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054808, upper bound: 0.0054806
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054398, upper bound: 0.0054410
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054398, upper bound: 0.0054409
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054806, upper bound: 0.0054798
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054806, upper bound: 0.0054830
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054773, upper bound: 0.0054814
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054798, upper bound: 0.0054779
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054731, upper bound: 0.0054708
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054731, upper bound: 0.0054696
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054846, upper bound: 0.0054802
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054846, upper bound: 0.0054826
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054821, upper bound: 0.0054799
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054825, upper bound: 0.0054793
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054793, upper bound: 0.0054799
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054808, upper bound: 0.0054766
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054649, upper bound: 0.0054823
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054840, upper bound: 0.0054619
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054832, upper bound: 0.0054825
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054844, upper bound: 0.0054793
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054836, upper bound: 0.0054809
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054838, upper bound: 0.0054830
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054816, upper bound: 0.0054795
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 17.39
Output dim: 3, lower bound: -0.0054816, upper bound: 0.0054831

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1553936, 0.1551163
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1636381, 0.1635064
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0308531, 0.0308418
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091227, 0.0091251
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0574044, 0.0574128
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0176556, 0.0176609
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612571, 0.0612987
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0383727, 0.0383894
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1622456, 0.1620497
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1466130, 0.1464649

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 933

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2257

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054757, upper bound: 0.0054788
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054757, upper bound: 0.0054792
time: 3.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1553936, 0.1551163
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1636381, 0.1635064
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0308531, 0.0308418
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091227, 0.0091251
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0574044, 0.0574128
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0176556, 0.0176609
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612571, 0.0612987
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0383727, 0.0383894
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1622456, 0.1620497
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1466130, 0.1464649

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2093

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054753, upper bound: 0.0054784
time: 4.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054756, upper bound: 0.0054783
time: 4.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1553933, 0.1551133
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1636392, 0.1635042
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0308530, 0.0308417
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091226, 0.0091251
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0574046, 0.0574122
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0176546, 0.0176605
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612568, 0.0612984
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0383719, 0.0383886
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1622442, 0.1620474
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1466132, 0.1464621

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2539

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2069

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054773, upper bound: 0.0054799
time: 15.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054773, upper bound: 0.0054803
time: 12.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1553906, 0.1551160
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1636360, 0.1635074
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0308530, 0.0308417
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091227, 0.0091250
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0574038, 0.0574130
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0176550, 0.0176600
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612568, 0.0612984
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0383720, 0.0383886
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1622433, 0.1620483
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1466102, 0.1464651

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 884

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2257

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054756, upper bound: 0.0054768
time: 11.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054760, upper bound: 0.0054789
time: 2.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1547110, 0.1545275
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1630998, 0.1629250
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0307047, 0.0306979
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0090940, 0.0090980
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573209, 0.0573069
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0175947, 0.0176027
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612215, 0.0612660
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0383724, 0.0383586
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1615564, 0.1615780
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1463779, 0.1462950

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 844

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054712, upper bound: 0.0054861
time: 8.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054712, upper bound: 0.0054852
time: 12.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1547017, 0.1545368
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1630899, 0.1629349
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0307114, 0.0306912
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0090953, 0.0090967
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573207, 0.0573071
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0175966, 0.0176008
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612229, 0.0612646
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0383722, 0.0383587
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1615396, 0.1615949
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1463673, 0.1463055

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 844

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054727, upper bound: 0.0054845
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054727, upper bound: 0.0054799
time: 22.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1548290, 0.1547835
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1632263, 0.1631633
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0308444, 0.0308321
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091205, 0.0091229
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573168, 0.0573140
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0176514, 0.0176573
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612536, 0.0612947
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0383722, 0.0383612
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1618612, 0.1619331
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1465595, 0.1464907

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2442

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054739, upper bound: 0.0054838
time: 10.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054739, upper bound: 0.0054832
time: 5.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1548290, 0.1547835
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1632263, 0.1631633
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0308444, 0.0308321
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091205, 0.0091229
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573168, 0.0573140
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0176514, 0.0176573
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612536, 0.0612947
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0383722, 0.0383612
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1618612, 0.1619331
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1465595, 0.1464907

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2293

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 989

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054730, upper bound: 0.0054838
time: 28.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054730, upper bound: 0.0054843
time: 2.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1553243, 0.1552197
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1641243, 0.1640968
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0308887, 0.0308805
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091234, 0.0091237
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0575214, 0.0575221
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0176848, 0.0176843
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612840, 0.0613262
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385571, 0.0385526
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1627347, 0.1628198
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1469384, 0.1469653

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2293

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054809, upper bound: 0.0054794
time: 10.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054815, upper bound: 0.0054820
time: 3.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1553175, 0.1552264
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1641174, 0.1641037
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0308885, 0.0308806
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091232, 0.0091238
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0575214, 0.0575221
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0176851, 0.0176841
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612840, 0.0613263
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385570, 0.0385526
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1627255, 0.1628290
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1469329, 0.1469709

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2069

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054806, upper bound: 0.0054829
time: 3.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054806, upper bound: 0.0054814
time: 54.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1555865, 0.1554402
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1643511, 0.1642954
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0308932, 0.0308852
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091264, 0.0091267
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0575129, 0.0575101
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0176986, 0.0176992
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612862, 0.0613282
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385944, 0.0385898
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1634301, 0.1635091
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1472882, 0.1473204

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 976

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3578

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054793, upper bound: 0.0054790
time: 5.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054817, upper bound: 0.0054807
time: 3.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1555237, 0.1555027
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1643111, 0.1643353
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0308933, 0.0308851
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0091264, 0.0091267
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0575094, 0.0575136
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0176996, 0.0176982
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0612861, 0.0613283
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385945, 0.0385898
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1633775, 0.1635616
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1472713, 0.1473373

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 686

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 989

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 2300

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054816, upper bound: 0.0054812
time: 189.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054816, upper bound: 0.0054824
time: 3.04 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 200.83 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 200.83
Output dim: 3, lower bound: -0.0054757, upper bound: 0.0054788
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 200.83
Output dim: 3, lower bound: -0.0054757, upper bound: 0.0054792
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 200.83
Output dim: 3, lower bound: -0.0054753, upper bound: 0.0054784
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 200.83
Output dim: 3, lower bound: -0.0054756, upper bound: 0.0054783
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 200.83
Output dim: 3, lower bound: -0.0054773, upper bound: 0.0054799
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 200.83
Output dim: 3, lower bound: -0.0054773, upper bound: 0.0054803
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 200.83
Output dim: 3, lower bound: -0.0054756, upper bound: 0.0054768
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 200.83
Output dim: 3, lower bound: -0.0054760, upper bound: 0.0054789
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 200.83
Output dim: 3, lower bound: -0.0054712, upper bound: 0.0054861
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 200.83
Output dim: 3, lower bound: -0.0054712, upper bound: 0.0054852
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 200.83
Output dim: 3, lower bound: -0.0054727, upper bound: 0.0054845
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 200.83
Output dim: 3, lower bound: -0.0054727, upper bound: 0.0054799
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 200.83
Output dim: 3, lower bound: -0.0054739, upper bound: 0.0054838
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 200.83
Output dim: 3, lower bound: -0.0054739, upper bound: 0.0054832
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 200.83
Output dim: 3, lower bound: -0.0054730, upper bound: 0.0054838
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 200.83
Output dim: 3, lower bound: -0.0054730, upper bound: 0.0054843
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 200.83
Output dim: 3, lower bound: -0.0054809, upper bound: 0.0054794
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 200.83
Output dim: 3, lower bound: -0.0054815, upper bound: 0.0054820
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 200.83
Output dim: 3, lower bound: -0.0054806, upper bound: 0.0054829
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 200.83
Output dim: 3, lower bound: -0.0054806, upper bound: 0.0054814
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 200.83
Output dim: 3, lower bound: -0.0054793, upper bound: 0.0054790
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 200.83
Output dim: 3, lower bound: -0.0054817, upper bound: 0.0054807
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 200.83
Output dim: 3, lower bound: -0.0054816, upper bound: 0.0054812
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 200.83
Output dim: 3, lower bound: -0.0054816, upper bound: 0.0054824
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 200.83
Output dim: 3, lower bound: -0.0054773, upper bound: 0.0054814
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 200.83
Output dim: 3, lower bound: -0.0054798, upper bound: 0.0054779
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 200.83
Output dim: 3, lower bound: -0.0054846, upper bound: 0.0054802
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 200.83
Output dim: 3, lower bound: -0.0054846, upper bound: 0.0054826
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 200.83
Output dim: 3, lower bound: -0.0054821, upper bound: 0.0054799
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 200.83
Output dim: 3, lower bound: -0.0054825, upper bound: 0.0054793
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 200.83
Output dim: 3, lower bound: -0.0054793, upper bound: 0.0054799
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 200.83
Output dim: 3, lower bound: -0.0054808, upper bound: 0.0054766
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 200.83
Output dim: 3, lower bound: -0.0054649, upper bound: 0.0054823
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 200.83
Output dim: 3, lower bound: -0.0054840, upper bound: 0.0054619
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 200.83
Output dim: 3, lower bound: -0.0054832, upper bound: 0.0054825
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 200.83
Output dim: 3, lower bound: -0.0054844, upper bound: 0.0054793
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 200.83
Output dim: 3, lower bound: -0.0054836, upper bound: 0.0054809
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 200.83
Output dim: 3, lower bound: -0.0054838, upper bound: 0.0054830
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 200.83
Output dim: 3, lower bound: -0.0054816, upper bound: 0.0054795
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 200.83
Output dim: 3, lower bound: -0.0054816, upper bound: 0.0054831

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 47.34 + 1840.43 = 1887.77 seconds
