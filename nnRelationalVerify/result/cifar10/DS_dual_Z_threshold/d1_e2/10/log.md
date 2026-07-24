## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 10)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0529606863


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.7559090, -4.1569338, -4.7559090, -4.1569338, -0.2038702, 0.2038702)
1: (-4.7411447, -4.2113256, -4.7411447, -4.2113256, -0.2005739, 0.2005739)
2: (-1.2580136, -1.0103903, -1.2580136, -1.0103903, -0.0831146, 0.0831146)
3: (-0.0328050, 0.3108093, -0.0328050, 0.3108093, -0.2911189, 0.2911189)
4: (-0.9833181, -0.6530223, -0.9833181, -0.6530223, -0.1150007, 0.1150007)
5: (-0.1361374, 0.3287530, -0.1361374, 0.3287530, -0.3120937, 0.3120938)
6: (0.5845805, 0.8929586, 0.5845805, 0.8929586, -0.1627387, 0.1627387)
7: (-1.1971323, -0.7447776, -1.1971323, -0.7447776, -0.0956259, 0.0956259)
8: (-5.1874294, -4.6035657, -5.1874294, -4.6035657, -0.2350359, 0.2350360)
9: (-5.3642797, -4.8095503, -5.3642797, -4.8095503, -0.2378700, 0.2378700)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.11 + 171.27 = 178.38 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0530126, upper bound: 0.0530138

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2798
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 3486
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 397
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 415
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 432
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3590

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3020

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0530099, upper bound: 0.0530124
time: 78.15 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0530117, upper bound: 0.0530103
time: 130.34 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 208.56 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 208.56
Output dim: 5, lower bound: -0.0530099, upper bound: 0.0530124
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 208.56
Output dim: 5, lower bound: -0.0530117, upper bound: 0.0530103

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -4.7559090, -4.1569338, -4.7559090, -4.1569338, -0.2038694, 0.2038694
1: -4.7411447, -4.2113256, -4.7411447, -4.2113256, -0.2005797, 0.2005798
2: -1.2580136, -1.0103903, -1.2580136, -1.0103903, -0.0830722, 0.0830704
3: -0.0328050, 0.3108093, -0.0328050, 0.3108093, -0.2911192, 0.2911192
4: -0.9833181, -0.6530223, -0.9833181, -0.6530223, -0.1150050, 0.1150047
5: -0.1361374, 0.3287530, -0.1361374, 0.3287530, -0.3121004, 0.3121005
6: 0.5845805, 0.8929586, 0.5845805, 0.8929586, -0.1627033, 0.1627042
7: -1.1971323, -0.7447776, -1.1971323, -0.7447776, -0.0955358, 0.0955362
8: -5.1874294, -4.6035657, -5.1874294, -4.6035657, -0.2350487, 0.2350488
9: -5.3642797, -4.8095503, -5.3642797, -4.8095503, -0.2378740, 0.2378742

Time for backsubstitution: 5.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2798
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 3486
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 397
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 415
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 432
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3590

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2572

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0529702, upper bound: 0.0529459
time: 183.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0529437, upper bound: 0.0529741
time: 43.33 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -4.7559090, -4.1569338, -4.7559090, -4.1569338, -0.2038694, 0.2038694
1: -4.7411447, -4.2113256, -4.7411447, -4.2113256, -0.2005798, 0.2005798
2: -1.2580136, -1.0103903, -1.2580136, -1.0103903, -0.0830704, 0.0830722
3: -0.0328050, 0.3108093, -0.0328050, 0.3108093, -0.2911192, 0.2911192
4: -0.9833181, -0.6530223, -0.9833181, -0.6530223, -0.1150047, 0.1150050
5: -0.1361374, 0.3287530, -0.1361374, 0.3287530, -0.3121005, 0.3121004
6: 0.5845805, 0.8929586, 0.5845805, 0.8929586, -0.1627042, 0.1627033
7: -1.1971323, -0.7447776, -1.1971323, -0.7447776, -0.0955362, 0.0955358
8: -5.1874294, -4.6035657, -5.1874294, -4.6035657, -0.2350488, 0.2350487
9: -5.3642797, -4.8095503, -5.3642797, -4.8095503, -0.2378742, 0.2378739

Time for backsubstitution: 5.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2798
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 3486
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 397
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 415
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 432
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3590

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2572

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0529720, upper bound: 0.0529446
time: 6.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0529446, upper bound: 0.0529740
time: 8.03 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 20.44 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 20.44
Output dim: 5, lower bound: -0.0529702, upper bound: 0.0529459
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 20.44
Output dim: 5, lower bound: -0.0529437, upper bound: 0.0529741
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 20.44
Output dim: 5, lower bound: -0.0529720, upper bound: 0.0529446
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 20.44
Output dim: 5, lower bound: -0.0529446, upper bound: 0.0529740

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7559090, -4.1569338, -4.7559090, -4.1569338, -0.2009911, 0.2011454
1: -4.7411447, -4.2113256, -4.7411447, -4.2113256, -0.1981786, 0.1983064
2: -1.2580136, -1.0103903, -1.2580136, -1.0103903, -0.0829362, 0.0829287
3: -0.0328050, 0.3108093, -0.0328050, 0.3108093, -0.2910299, 0.2910261
4: -0.9833181, -0.6530223, -0.9833181, -0.6530223, -0.1140073, 0.1139902
5: -0.1361374, 0.3287530, -0.1361374, 0.3287530, -0.3118489, 0.3118446
6: 0.5845805, 0.8929586, 0.5845805, 0.8929586, -0.1624913, 0.1624920
7: -1.1971323, -0.7447776, -1.1971323, -0.7447776, -0.0953547, 0.0953526
8: -5.1874294, -4.6035657, -5.1874294, -4.6035657, -0.2341921, 0.2342083
9: -5.3642797, -4.8095503, -5.3642797, -4.8095503, -0.2372012, 0.2372144

Time for backsubstitution: 5.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2798
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 3486
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 397
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 415
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 432
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3590

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0529353, upper bound: 0.0528978
time: 112.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0529218, upper bound: 0.0529125
time: 49.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.7559090, -4.1569338, -4.7559090, -4.1569338, -0.2011453, 0.2009911
1: -4.7411447, -4.2113256, -4.7411447, -4.2113256, -0.1983063, 0.1981787
2: -1.2580136, -1.0103903, -1.2580136, -1.0103903, -0.0829305, 0.0829344
3: -0.0328050, 0.3108093, -0.0328050, 0.3108093, -0.2910261, 0.2910298
4: -0.9833181, -0.6530223, -0.9833181, -0.6530223, -0.1139905, 0.1140071
5: -0.1361374, 0.3287530, -0.1361374, 0.3287530, -0.3118444, 0.3118490
6: 0.5845805, 0.8929586, 0.5845805, 0.8929586, -0.1624911, 0.1624922
7: -1.1971323, -0.7447776, -1.1971323, -0.7447776, -0.0953523, 0.0953550
8: -5.1874294, -4.6035657, -5.1874294, -4.6035657, -0.2342081, 0.2341923
9: -5.3642797, -4.8095503, -5.3642797, -4.8095503, -0.2372143, 0.2372015

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2798
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 3486
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 397
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 415
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 432
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3590

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2134

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0529085, upper bound: 0.0529245
time: 289.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0528943, upper bound: 0.0529387
time: 144.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.7559090, -4.1569338, -4.7559090, -4.1569338, -0.2009911, 0.2011454
1: -4.7411447, -4.2113256, -4.7411447, -4.2113256, -0.1981787, 0.1983063
2: -1.2580136, -1.0103903, -1.2580136, -1.0103903, -0.0829344, 0.0829305
3: -0.0328050, 0.3108093, -0.0328050, 0.3108093, -0.2910299, 0.2910261
4: -0.9833181, -0.6530223, -0.9833181, -0.6530223, -0.1140071, 0.1139905
5: -0.1361374, 0.3287530, -0.1361374, 0.3287530, -0.3118490, 0.3118445
6: 0.5845805, 0.8929586, 0.5845805, 0.8929586, -0.1624922, 0.1624911
7: -1.1971323, -0.7447776, -1.1971323, -0.7447776, -0.0953550, 0.0953523
8: -5.1874294, -4.6035657, -5.1874294, -4.6035657, -0.2341923, 0.2342081
9: -5.3642797, -4.8095503, -5.3642797, -4.8095503, -0.2372014, 0.2372143

Time for backsubstitution: 5.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2798
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 3486
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 397
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 415
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 432
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3590

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2134

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0529376, upper bound: 0.0528958
time: 126.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0529250, upper bound: 0.0529082
time: 205.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.7559090, -4.1569338, -4.7559090, -4.1569338, -0.2011453, 0.2009911
1: -4.7411447, -4.2113256, -4.7411447, -4.2113256, -0.1983064, 0.1981786
2: -1.2580136, -1.0103903, -1.2580136, -1.0103903, -0.0829287, 0.0829362
3: -0.0328050, 0.3108093, -0.0328050, 0.3108093, -0.2910262, 0.2910298
4: -0.9833181, -0.6530223, -0.9833181, -0.6530223, -0.1139902, 0.1140073
5: -0.1361374, 0.3287530, -0.1361374, 0.3287530, -0.3118446, 0.3118489
6: 0.5845805, 0.8929586, 0.5845805, 0.8929586, -0.1624920, 0.1624913
7: -1.1971323, -0.7447776, -1.1971323, -0.7447776, -0.0953526, 0.0953547
8: -5.1874294, -4.6035657, -5.1874294, -4.6035657, -0.2342082, 0.2341922
9: -5.3642797, -4.8095503, -5.3642797, -4.8095503, -0.2372145, 0.2372012

Time for backsubstitution: 5.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2798
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 3486
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 397
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 415
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 432
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3590

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2134

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0529099, upper bound: 0.0529206
time: 115.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0528977, upper bound: 0.0529372
time: 148.59 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 269.89 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 269.89
Output dim: 5, lower bound: -0.0529353, upper bound: 0.0528978
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 269.89
Output dim: 5, lower bound: -0.0529218, upper bound: 0.0529125
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 269.89
Output dim: 5, lower bound: -0.0529085, upper bound: 0.0529245
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 269.89
Output dim: 5, lower bound: -0.0528943, upper bound: 0.0529387
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 269.89
Output dim: 5, lower bound: -0.0529376, upper bound: 0.0528958
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 269.89
Output dim: 5, lower bound: -0.0529250, upper bound: 0.0529082
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 269.89
Output dim: 5, lower bound: -0.0529099, upper bound: 0.0529206
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 269.89
Output dim: 5, lower bound: -0.0528977, upper bound: 0.0529372

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 178.38 + 1675.52 = 1853.90 seconds
