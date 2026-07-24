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
execution time: IAR + RelationalAnalysis = 7.10 + 39.06 = 46.16 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0054881, upper bound: 0.0054897

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2122
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2122

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054849, upper bound: 0.0054866
time: 10.06 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054857, upper bound: 0.0054867
time: 2.95 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 13.09 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 13.09
Output dim: 3, lower bound: -0.0054849, upper bound: 0.0054866
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 13.09
Output dim: 3, lower bound: -0.0054857, upper bound: 0.0054867

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1608026, 0.1607945
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1687119, 0.1686784
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0310469, 0.0310487
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092762, 0.0092779
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0574486, 0.0574515
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178793, 0.0178779
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622222, 0.0622212
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0386506, 0.0386504
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1671094, 0.1671032
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1494899, 0.1494774

Time for backsubstitution: 5.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2539

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054836, upper bound: 0.0054853
time: 2.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054809, upper bound: 0.0054846
time: 12.11 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1607945, 0.1608026
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1686784, 0.1687119
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0310487, 0.0310469
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092779, 0.0092762
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0574515, 0.0574486
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178779, 0.0178793
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622212, 0.0622222
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0386504, 0.0386506
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1671032, 0.1671094
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1494774, 0.1494899

Time for backsubstitution: 5.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2539

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054847, upper bound: 0.0054817
time: 14.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054828, upper bound: 0.0054833
time: 33.37 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 53.12 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 53.12
Output dim: 3, lower bound: -0.0054836, upper bound: 0.0054853
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 53.12
Output dim: 3, lower bound: -0.0054809, upper bound: 0.0054846
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 53.12
Output dim: 3, lower bound: -0.0054847, upper bound: 0.0054817
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 53.12
Output dim: 3, lower bound: -0.0054828, upper bound: 0.0054833

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1606644, 0.1607487
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1685693, 0.1686148
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0310404, 0.0310451
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092740, 0.0092747
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0574414, 0.0574454
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178754, 0.0178725
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622219, 0.0622208
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0386483, 0.0386481
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1670128, 0.1670400
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1493621, 0.1493913

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2557

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054821, upper bound: 0.0054829
time: 3.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054814, upper bound: 0.0054840
time: 2.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1607569, 0.1606562
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1686483, 0.1685358
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0310434, 0.0310421
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092729, 0.0092757
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0574426, 0.0574442
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178739, 0.0178740
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622219, 0.0622209
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0386483, 0.0386481
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1670461, 0.1670066
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1494039, 0.1493495

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2557

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054800, upper bound: 0.0054834
time: 3.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054794, upper bound: 0.0054854
time: 3.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1606562, 0.1607568
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1685358, 0.1686483
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0310421, 0.0310434
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092757, 0.0092729
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0574442, 0.0574426
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178740, 0.0178739
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622209, 0.0622219
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0386481, 0.0386483
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1670066, 0.1670461
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1493495, 0.1494039

Time for backsubstitution: 5.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2557

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054837, upper bound: 0.0054801
time: 22.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054823, upper bound: 0.0054820
time: 6.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1607487, 0.1606644
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1686148, 0.1685693
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0310451, 0.0310404
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092747, 0.0092740
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0574454, 0.0574414
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178725, 0.0178754
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622208, 0.0622219
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0386481, 0.0386483
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1670400, 0.1670128
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1493913, 0.1493621

Time for backsubstitution: 5.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2557
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2557

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054817, upper bound: 0.0054830
time: 9.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054803, upper bound: 0.0054802
time: 25.20 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 40.16 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 40.16
Output dim: 3, lower bound: -0.0054821, upper bound: 0.0054829
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 40.16
Output dim: 3, lower bound: -0.0054814, upper bound: 0.0054840
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 40.16
Output dim: 3, lower bound: -0.0054800, upper bound: 0.0054834
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 40.16
Output dim: 3, lower bound: -0.0054794, upper bound: 0.0054854
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 40.16
Output dim: 3, lower bound: -0.0054837, upper bound: 0.0054801
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 40.16
Output dim: 3, lower bound: -0.0054823, upper bound: 0.0054820
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 40.16
Output dim: 3, lower bound: -0.0054817, upper bound: 0.0054830
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 40.16
Output dim: 3, lower bound: -0.0054803, upper bound: 0.0054802

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1606781, 0.1607597
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1685749, 0.1686190
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0310466, 0.0310492
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092745, 0.0092751
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573903, 0.0573929
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178765, 0.0178736
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622109, 0.0622115
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385530, 0.0385507
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1669912, 0.1670187
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1493470, 0.1493759

Time for backsubstitution: 5.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054783, upper bound: 0.0054778
time: 10.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054783, upper bound: 0.0054776
time: 36.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1606754, 0.1607630
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1685734, 0.1686207
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0310444, 0.0310516
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092745, 0.0092751
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573888, 0.0573945
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178766, 0.0178736
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622126, 0.0622098
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385509, 0.0385528
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1669918, 0.1670184
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1493468, 0.1493762

Time for backsubstitution: 5.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054780, upper bound: 0.0054799
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054779, upper bound: 0.0054792
time: 6.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1607705, 0.1606672
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1686540, 0.1685399
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0310496, 0.0310462
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092734, 0.0092762
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573915, 0.0573917
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178750, 0.0178751
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622108, 0.0622115
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385530, 0.0385507
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1670246, 0.1669853
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1493888, 0.1493342

Time for backsubstitution: 5.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054763, upper bound: 0.0054799
time: 5.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054763, upper bound: 0.0054804
time: 2.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1607678, 0.1606705
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1686524, 0.1685417
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0310474, 0.0310486
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092734, 0.0092762
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573900, 0.0573933
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178750, 0.0178751
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622125, 0.0622098
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385509, 0.0385528
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1670252, 0.1669850
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1493886, 0.1493345

Time for backsubstitution: 5.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054757, upper bound: 0.0054795
time: 39.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054756, upper bound: 0.0054812
time: 4.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1606705, 0.1607679
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1685417, 0.1686524
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0310486, 0.0310474
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092762, 0.0092734
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573933, 0.0573901
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178751, 0.0178750
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622098, 0.0622125
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385528, 0.0385509
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1669850, 0.1670252
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1493345, 0.1493886

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2167

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054807, upper bound: 0.0054776
time: 3.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054803, upper bound: 0.0054785
time: 2.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1606672, 0.1607705
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1685399, 0.1686540
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0310462, 0.0310496
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092762, 0.0092734
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573917, 0.0573915
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178751, 0.0178750
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622115, 0.0622109
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385507, 0.0385530
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1669853, 0.1670246
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1493342, 0.1493888

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2167

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054786, upper bound: 0.0054776
time: 3.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054786, upper bound: 0.0054779
time: 8.34 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1607630, 0.1606754
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1686207, 0.1685734
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0310516, 0.0310444
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092751, 0.0092745
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573945, 0.0573888
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178736, 0.0178766
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622098, 0.0622125
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385528, 0.0385509
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1670184, 0.1669918
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1493762, 0.1493468

Time for backsubstitution: 5.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2167

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054782, upper bound: 0.0054775
time: 21.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054779, upper bound: 0.0054800
time: 2.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1607597, 0.1606781
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1686190, 0.1685749
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0310492, 0.0310466
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092751, 0.0092745
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573929, 0.0573903
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178736, 0.0178765
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622115, 0.0622109
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385507, 0.0385530
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1670187, 0.1669912
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1493759, 0.1493470

Time for backsubstitution: 5.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2167
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2167

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054766, upper bound: 0.0054793
time: 5.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054766, upper bound: 0.0054793
time: 23.61 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 34.45 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 34.45
Output dim: 3, lower bound: -0.0054783, upper bound: 0.0054778
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 34.45
Output dim: 3, lower bound: -0.0054783, upper bound: 0.0054776
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.45
Output dim: 3, lower bound: -0.0054780, upper bound: 0.0054799
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.45
Output dim: 3, lower bound: -0.0054779, upper bound: 0.0054792
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.45
Output dim: 3, lower bound: -0.0054763, upper bound: 0.0054799
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.45
Output dim: 3, lower bound: -0.0054763, upper bound: 0.0054804
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.45
Output dim: 3, lower bound: -0.0054757, upper bound: 0.0054795
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.45
Output dim: 3, lower bound: -0.0054756, upper bound: 0.0054812
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.45
Output dim: 3, lower bound: -0.0054807, upper bound: 0.0054776
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.45
Output dim: 3, lower bound: -0.0054803, upper bound: 0.0054785
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.45
Output dim: 3, lower bound: -0.0054786, upper bound: 0.0054776
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.45
Output dim: 3, lower bound: -0.0054786, upper bound: 0.0054779
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 34.45
Output dim: 3, lower bound: -0.0054782, upper bound: 0.0054775
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.45
Output dim: 3, lower bound: -0.0054779, upper bound: 0.0054800
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.45
Output dim: 3, lower bound: -0.0054766, upper bound: 0.0054793
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.45
Output dim: 3, lower bound: -0.0054766, upper bound: 0.0054793

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1604744, 0.1604552
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1683555, 0.1683758
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309888, 0.0309963
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092578, 0.0092583
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573507, 0.0573638
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178525, 0.0178495
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622013, 0.0622023
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385496, 0.0385515
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1669198, 0.1669003
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1491998, 0.1492261

Time for backsubstitution: 5.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 769

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054745, upper bound: 0.0054768
time: 9.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054765, upper bound: 0.0054755
time: 2.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1603676, 0.1605620
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1683285, 0.1684029
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309891, 0.0309960
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092576, 0.0092584
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573581, 0.0573564
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178525, 0.0178495
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622050, 0.0621986
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385496, 0.0385515
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1668737, 0.1669464
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1491967, 0.1492292

Time for backsubstitution: 5.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 769

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054744, upper bound: 0.0054779
time: 14.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054767, upper bound: 0.0054741
time: 40.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1605695, 0.1603594
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1684361, 0.1682950
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309939, 0.0309908
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092567, 0.0092593
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573534, 0.0573610
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178509, 0.0178511
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0621996, 0.0622040
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385517, 0.0385494
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1669526, 0.1668673
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1492417, 0.1491840

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 769

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054727, upper bound: 0.0054792
time: 9.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054763, upper bound: 0.0054751
time: 4.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1604627, 0.1604662
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1684091, 0.1683221
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309942, 0.0309905
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092566, 0.0092594
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573608, 0.0573535
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178509, 0.0178511
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622033, 0.0622003
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385517, 0.0385494
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1669065, 0.1669133
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1492387, 0.1491871

Time for backsubstitution: 5.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 769

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054728, upper bound: 0.0054788
time: 20.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054755, upper bound: 0.0054761
time: 105.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1605668, 0.1603628
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1684346, 0.1682968
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309918, 0.0309933
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092567, 0.0092594
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573519, 0.0573626
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178510, 0.0178511
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622013, 0.0622023
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385496, 0.0385515
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1669532, 0.1668670
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1492415, 0.1491843

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 769

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054728, upper bound: 0.0054806
time: 2.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054757, upper bound: 0.0054749
time: 33.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1604601, 0.1604695
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1684076, 0.1683238
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309921, 0.0309930
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092566, 0.0092595
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573593, 0.0573552
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178510, 0.0178510
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622050, 0.0621986
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385496, 0.0385515
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1669071, 0.1669130
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1492385, 0.1491874

Time for backsubstitution: 5.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 769

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054727, upper bound: 0.0054812
time: 3.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054755, upper bound: 0.0054763
time: 19.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1604695, 0.1604601
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1683238, 0.1684076
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309930, 0.0309921
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092595, 0.0092566
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573552, 0.0573593
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178510, 0.0178510
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0621986, 0.0622050
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385515, 0.0385496
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1669130, 0.1669071
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1491874, 0.1492384

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 769

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054775, upper bound: 0.0054730
time: 81.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054790, upper bound: 0.0054743
time: 2.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1603628, 0.1605668
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1682968, 0.1684346
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309933, 0.0309918
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092594, 0.0092567
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573626, 0.0573519
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178511, 0.0178510
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622023, 0.0622013
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385515, 0.0385496
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1668670, 0.1669531
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1491843, 0.1492415

Time for backsubstitution: 5.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 769

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054764, upper bound: 0.0054759
time: 4.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054789, upper bound: 0.0054729
time: 20.35 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1604662, 0.1604627
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1683221, 0.1684091
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309905, 0.0309942
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092594, 0.0092566
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573535, 0.0573608
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178511, 0.0178509
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622003, 0.0622033
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385494, 0.0385517
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1669133, 0.1669065
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1491871, 0.1492387

Time for backsubstitution: 5.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 769

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054749, upper bound: 0.0054772
time: 2.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054785, upper bound: 0.0054736
time: 11.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1603594, 0.1605695
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1682950, 0.1684361
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309908, 0.0309939
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092593, 0.0092567
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573610, 0.0573534
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178511, 0.0178509
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622040, 0.0621996
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385494, 0.0385517
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1668673, 0.1669526
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1491841, 0.1492417

Time for backsubstitution: 5.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 769

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054752, upper bound: 0.0054749
time: 8.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054775, upper bound: 0.0054747
time: 5.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1604552, 0.1604744
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1683759, 0.1683555
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309963, 0.0309888
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092583, 0.0092578
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573638, 0.0573507
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178495, 0.0178525
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622023, 0.0622013
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385515, 0.0385496
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1669004, 0.1669198
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1492261, 0.1491997

Time for backsubstitution: 5.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 769

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054745, upper bound: 0.0054786
time: 6.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054769, upper bound: 0.0054749
time: 6.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1605587, 0.1603703
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1684011, 0.1683300
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309935, 0.0309912
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092584, 0.0092577
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573547, 0.0573596
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178496, 0.0178525
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622003, 0.0622033
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385494, 0.0385517
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1669467, 0.1668731
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1492289, 0.1491969

Time for backsubstitution: 5.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 769

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054738, upper bound: 0.0054783
time: 15.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054757, upper bound: 0.0054742
time: 32.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1604519, 0.1604770
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1683741, 0.1683570
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309938, 0.0309909
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092583, 0.0092578
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573622, 0.0573522
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0178496, 0.0178525
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0622039, 0.0621996
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385494, 0.0385517
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1669007, 0.1669192
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1492258, 0.1492000

Time for backsubstitution: 5.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 769
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 769

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0054738, upper bound: 0.0054792
time: 7.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054756, upper bound: 0.0054742
time: 7.08 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 20.04 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054745, upper bound: 0.0054768
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054765, upper bound: 0.0054755
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054744, upper bound: 0.0054779
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054767, upper bound: 0.0054741
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054727, upper bound: 0.0054792
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054763, upper bound: 0.0054751
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054728, upper bound: 0.0054788
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054755, upper bound: 0.0054761
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054728, upper bound: 0.0054806
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054757, upper bound: 0.0054749
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054727, upper bound: 0.0054812
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054755, upper bound: 0.0054763
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054775, upper bound: 0.0054730
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054790, upper bound: 0.0054743
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054764, upper bound: 0.0054759
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054789, upper bound: 0.0054729
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054749, upper bound: 0.0054772
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054785, upper bound: 0.0054736
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054752, upper bound: 0.0054749
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054775, upper bound: 0.0054747
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054745, upper bound: 0.0054786
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054769, upper bound: 0.0054749
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054738, upper bound: 0.0054783
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054757, upper bound: 0.0054742
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054738, upper bound: 0.0054792
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 20.04
Output dim: 3, lower bound: -0.0054756, upper bound: 0.0054742

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1596023, 0.1593503
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1674320, 0.1672403
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309184, 0.0309128
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092164, 0.0092206
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573164, 0.0573241
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177946, 0.0177963
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0621886, 0.0621931
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385198, 0.0385171
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1663129, 0.1662030
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1484740, 0.1483797

Time for backsubstitution: 5.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3039

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054619, upper bound: 0.0054677
time: 19.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054617, upper bound: 0.0054661
time: 32.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1594955, 0.1594571
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1674050, 0.1672673
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309187, 0.0309125
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092163, 0.0092208
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573239, 0.0573166
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177946, 0.0177963
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0621923, 0.0621894
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385199, 0.0385171
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1662669, 0.1662491
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1484709, 0.1483828

Time for backsubstitution: 5.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3039

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054623, upper bound: 0.0054678
time: 20.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054620, upper bound: 0.0054649
time: 37.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1595996, 0.1593536
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1674305, 0.1672421
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309162, 0.0309153
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092163, 0.0092207
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573149, 0.0573257
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177947, 0.0177963
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0621903, 0.0621914
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385178, 0.0385192
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1663135, 0.1662027
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1484737, 0.1483800

Time for backsubstitution: 5.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3039

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054611, upper bound: 0.0054699
time: 3.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054606, upper bound: 0.0054636
time: 36.03 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1594928, 0.1594604
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1674035, 0.1672691
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309165, 0.0309150
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092162, 0.0092208
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573224, 0.0573183
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177947, 0.0177963
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0621940, 0.0621877
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385178, 0.0385191
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1662675, 0.1662487
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1484707, 0.1483831

Time for backsubstitution: 5.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3039

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054611, upper bound: 0.0054692
time: 3.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054616, upper bound: 0.0054683
time: 4.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1594604, 0.1594928
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1672691, 0.1674035
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309150, 0.0309165
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092208, 0.0092162
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573183, 0.0573224
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177963, 0.0177947
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0621877, 0.0621940
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385191, 0.0385178
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1662487, 0.1662675
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1483831, 0.1484707

Time for backsubstitution: 5.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3039

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054683, upper bound: 0.0054622
time: 13.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054676, upper bound: 0.0054627
time: 8.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1593536, 0.1595996
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1672421, 0.1674305
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309153, 0.0309162
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092207, 0.0092163
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573257, 0.0573150
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177963, 0.0177947
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0621914, 0.0621903
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385192, 0.0385178
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1662027, 0.1663135
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1483800, 0.1484737

Time for backsubstitution: 5.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 3039

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054676, upper bound: 0.0054617
time: 10.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054677, upper bound: 0.0054627
time: 20.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1594880, 0.1594652
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1673718, 0.1673008
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309207, 0.0309108
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092180, 0.0092191
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573269, 0.0573138
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177932, 0.0177977
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0621913, 0.0621904
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385197, 0.0385173
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1662607, 0.1662555
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1484583, 0.1483954

Time for backsubstitution: 5.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3039

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054628, upper bound: 0.0054665
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054639, upper bound: 0.0054668
time: 4.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.0889463, -5.3182545, -6.0889463, -5.3182545, -0.1594847, 0.1594679
1: -4.1810565, -3.4585633, -4.1810565, -3.4585633, -0.1673700, 0.1673023
2: -2.4615386, -2.2315230, -2.4615386, -2.2315230, -0.0309183, 0.0309129
3: 0.0919671, 0.1948055, 0.0919671, 0.1948055, -0.0092179, 0.0092191
4: -1.3965774, -1.0505964, -1.3965774, -1.0505964, -0.0573252, 0.0573153
5: 0.4018837, 0.5862499, 0.4018837, 0.5862499, -0.0177933, 0.0177977
6: -1.2454917, -1.0264900, -1.2454917, -1.0264900, -0.0621930, 0.0621888
7: 0.1242563, 0.4758165, 0.1242563, 0.4758165, -0.0385176, 0.0385193
8: -3.7379642, -2.8694506, -3.7379642, -2.8694506, -0.1662610, 0.1662549
9: -4.5816584, -3.9145586, -4.5816584, -3.9145586, -0.1484580, 0.1483956

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2348
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3123
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2304
type: DSZ, layer: 1, pos: 2291
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 857
type: DSZ, layer: 1, pos: 40
type: DSZ, layer: 1, pos: 2300
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 169
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2196
type: DSZ, layer: 1, pos: 2648
type: DSZ, layer: 1, pos: 338
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 727
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 548
type: DSZ, layer: 1, pos: 875
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 488
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 326
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 702
type: DSZ, layer: 1, pos: 765
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 808
type: DSZ, layer: 1, pos: 837
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 878
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 892
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 898
type: DSZ, layer: 1, pos: 903
type: DSZ, layer: 1, pos: 905
type: DSZ, layer: 1, pos: 918
type: DSZ, layer: 1, pos: 932
type: DSZ, layer: 1, pos: 933
type: DSZ, layer: 1, pos: 947
type: DSZ, layer: 1, pos: 948
type: DSZ, layer: 1, pos: 962
type: DSZ, layer: 1, pos: 963
type: DSZ, layer: 1, pos: 976
type: DSZ, layer: 1, pos: 977
type: DSZ, layer: 1, pos: 989
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2069
type: DSZ, layer: 1, pos: 2093
type: DSZ, layer: 1, pos: 2113
type: DSZ, layer: 1, pos: 2116
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2158
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2240
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2257
type: DSZ, layer: 1, pos: 2262
type: DSZ, layer: 1, pos: 2263
type: DSZ, layer: 1, pos: 2292
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2307
type: DSZ, layer: 1, pos: 2442
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2610
type: DSZ, layer: 1, pos: 2612
type: DSZ, layer: 1, pos: 2676
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2932
type: DSZ, layer: 1, pos: 2984
type: DSZ, layer: 1, pos: 2996
type: DSZ, layer: 1, pos: 3274
type: DSZ, layer: 1, pos: 3578

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3039

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054624, upper bound: 0.0054673
time: 3.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0054619, upper bound: 0.0054668
time: 3.93 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 13.01 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 13.01
Output dim: 3, lower bound: -0.0054619, upper bound: 0.0054677
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 13.01
Output dim: 3, lower bound: -0.0054617, upper bound: 0.0054661
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 13.01
Output dim: 3, lower bound: -0.0054623, upper bound: 0.0054678
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 13.01
Output dim: 3, lower bound: -0.0054620, upper bound: 0.0054649
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 13.01
Output dim: 3, lower bound: -0.0054611, upper bound: 0.0054699
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 13.01
Output dim: 3, lower bound: -0.0054606, upper bound: 0.0054636
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 13.01
Output dim: 3, lower bound: -0.0054611, upper bound: 0.0054692
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 13.01
Output dim: 3, lower bound: -0.0054616, upper bound: 0.0054683
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 13.01
Output dim: 3, lower bound: -0.0054683, upper bound: 0.0054622
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 13.01
Output dim: 3, lower bound: -0.0054676, upper bound: 0.0054627
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 13.01
Output dim: 3, lower bound: -0.0054676, upper bound: 0.0054617
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 13.01
Output dim: 3, lower bound: -0.0054677, upper bound: 0.0054627
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 13.01
Output dim: 3, lower bound: -0.0054628, upper bound: 0.0054665
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 13.01
Output dim: 3, lower bound: -0.0054639, upper bound: 0.0054668
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 13.01
Output dim: 3, lower bound: -0.0054624, upper bound: 0.0054673
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 13.01
Output dim: 3, lower bound: -0.0054619, upper bound: 0.0054668

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 46.16 + 1228.77 = 1274.93 seconds
