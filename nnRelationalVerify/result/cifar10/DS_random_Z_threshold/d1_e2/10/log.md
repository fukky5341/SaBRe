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
execution time: IAR + RelationalAnalysis = 8.29 + 168.42 = 176.71 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0530126, upper bound: 0.0530138

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2892
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 3486
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 397
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 415
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 432
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2798
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2371

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2892

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0530120, upper bound: 0.0530112
time: 119.92 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0530106, upper bound: 0.0530135
time: 16.49 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 136.42 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 136.42
Output dim: 5, lower bound: -0.0530120, upper bound: 0.0530112
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 136.42
Output dim: 5, lower bound: -0.0530106, upper bound: 0.0530135

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -4.7559090, -4.1569338, -4.7559090, -4.1569338, -0.2037835, 0.2037794
1: -4.7411447, -4.2113256, -4.7411447, -4.2113256, -0.2005285, 0.2005285
2: -1.2580136, -1.0103903, -1.2580136, -1.0103903, -0.0831138, 0.0831138
3: -0.0328050, 0.3108093, -0.0328050, 0.3108093, -0.2911188, 0.2911187
4: -0.9833181, -0.6530223, -0.9833181, -0.6530223, -0.1149865, 0.1149868
5: -0.1361374, 0.3287530, -0.1361374, 0.3287530, -0.3120938, 0.3120938
6: 0.5845805, 0.8929586, 0.5845805, 0.8929586, -0.1627383, 0.1627383
7: -1.1971323, -0.7447776, -1.1971323, -0.7447776, -0.0955892, 0.0955878
8: -5.1874294, -4.6035657, -5.1874294, -4.6035657, -0.2349847, 0.2349833
9: -5.3642797, -4.8095503, -5.3642797, -4.8095503, -0.2378509, 0.2378512

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 397
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 415
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2798
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 432
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 3486
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 3322

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2081

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0530061, upper bound: 0.0530111
time: 119.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0530112, upper bound: 0.0530078
time: 15.89 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -4.7559090, -4.1569338, -4.7559090, -4.1569338, -0.2037794, 0.2037834
1: -4.7411447, -4.2113256, -4.7411447, -4.2113256, -0.2005285, 0.2005285
2: -1.2580136, -1.0103903, -1.2580136, -1.0103903, -0.0831138, 0.0831138
3: -0.0328050, 0.3108093, -0.0328050, 0.3108093, -0.2911188, 0.2911187
4: -0.9833181, -0.6530223, -0.9833181, -0.6530223, -0.1149868, 0.1149865
5: -0.1361374, 0.3287530, -0.1361374, 0.3287530, -0.3120938, 0.3120938
6: 0.5845805, 0.8929586, 0.5845805, 0.8929586, -0.1627383, 0.1627383
7: -1.1971323, -0.7447776, -1.1971323, -0.7447776, -0.0955878, 0.0955892
8: -5.1874294, -4.6035657, -5.1874294, -4.6035657, -0.2349834, 0.2349846
9: -5.3642797, -4.8095503, -5.3642797, -4.8095503, -0.2378512, 0.2378509

Time for backsubstitution: 6.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 432
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 415
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2798
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 397
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 3486
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2572

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2162

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0529941, upper bound: 0.0529995
time: 13.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0529972, upper bound: 0.0529962
time: 118.75 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 138.87 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 138.87
Output dim: 5, lower bound: -0.0530061, upper bound: 0.0530111
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 138.87
Output dim: 5, lower bound: -0.0530112, upper bound: 0.0530078
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 138.87
Output dim: 5, lower bound: -0.0529941, upper bound: 0.0529995
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 138.87
Output dim: 5, lower bound: -0.0529972, upper bound: 0.0529962

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7559090, -4.1569338, -4.7559090, -4.1569338, -0.2032362, 0.2031952
1: -4.7411447, -4.2113256, -4.7411447, -4.2113256, -0.1999552, 0.1999256
2: -1.2580136, -1.0103903, -1.2580136, -1.0103903, -0.0831265, 0.0831255
3: -0.0328050, 0.3108093, -0.0328050, 0.3108093, -0.2911204, 0.2911204
4: -0.9833181, -0.6530223, -0.9833181, -0.6530223, -0.1149057, 0.1149001
5: -0.1361374, 0.3287530, -0.1361374, 0.3287530, -0.3121209, 0.3121210
6: 0.5845805, 0.8929586, 0.5845805, 0.8929586, -0.1626756, 0.1626776
7: -1.1971323, -0.7447776, -1.1971323, -0.7447776, -0.0954747, 0.0954679
8: -5.1874294, -4.6035657, -5.1874294, -4.6035657, -0.2346875, 0.2346638
9: -5.3642797, -4.8095503, -5.3642797, -4.8095503, -0.2377025, 0.2377025

Time for backsubstitution: 6.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 3486
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 397
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 415
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 432
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2798
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 3366

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3117

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0530055, upper bound: 0.0530119
time: 17.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0530053, upper bound: 0.0530098
time: 114.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.7559090, -4.1569338, -4.7559090, -4.1569338, -0.2031992, 0.2032322
1: -4.7411447, -4.2113256, -4.7411447, -4.2113256, -0.1999256, 0.1999551
2: -1.2580136, -1.0103903, -1.2580136, -1.0103903, -0.0831255, 0.0831265
3: -0.0328050, 0.3108093, -0.0328050, 0.3108093, -0.2911204, 0.2911204
4: -0.9833181, -0.6530223, -0.9833181, -0.6530223, -0.1148998, 0.1149060
5: -0.1361374, 0.3287530, -0.1361374, 0.3287530, -0.3121210, 0.3121209
6: 0.5845805, 0.8929586, 0.5845805, 0.8929586, -0.1626775, 0.1626756
7: -1.1971323, -0.7447776, -1.1971323, -0.7447776, -0.0954693, 0.0954733
8: -5.1874294, -4.6035657, -5.1874294, -4.6035657, -0.2346651, 0.2346862
9: -5.3642797, -4.8095503, -5.3642797, -4.8095503, -0.2377023, 0.2377027

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3486
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 432
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 415
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 397
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2798
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2825

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2395

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0529389, upper bound: 0.0529847
time: 101.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0529905, upper bound: 0.0529363
time: 6.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.7559090, -4.1569338, -4.7559090, -4.1569338, -0.2032278, 0.2031862
1: -4.7411447, -4.2113256, -4.7411447, -4.2113256, -0.1995230, 0.1994753
2: -1.2580136, -1.0103903, -1.2580136, -1.0103903, -0.0831031, 0.0831026
3: -0.0328050, 0.3108093, -0.0328050, 0.3108093, -0.2911182, 0.2911181
4: -0.9833181, -0.6530223, -0.9833181, -0.6530223, -0.1149867, 0.1149864
5: -0.1361374, 0.3287530, -0.1361374, 0.3287530, -0.3120772, 0.3120778
6: 0.5845805, 0.8929586, 0.5845805, 0.8929586, -0.1627065, 0.1627083
7: -1.1971323, -0.7447776, -1.1971323, -0.7447776, -0.0953192, 0.0953134
8: -5.1874294, -4.6035657, -5.1874294, -4.6035657, -0.2337053, 0.2336441
9: -5.3642797, -4.8095503, -5.3642797, -4.8095503, -0.2366542, 0.2366057

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 3486
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 397
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 432
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 415
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2798
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 3531

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2211

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0529903, upper bound: 0.0529947
time: 505.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0529898, upper bound: 0.0529962
time: 76.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.7559090, -4.1569338, -4.7559090, -4.1569338, -0.2031822, 0.2032318
1: -4.7411447, -4.2113256, -4.7411447, -4.2113256, -0.1994753, 0.1995231
2: -1.2580136, -1.0103903, -1.2580136, -1.0103903, -0.0831026, 0.0831031
3: -0.0328050, 0.3108093, -0.0328050, 0.3108093, -0.2911182, 0.2911181
4: -0.9833181, -0.6530223, -0.9833181, -0.6530223, -0.1149867, 0.1149864
5: -0.1361374, 0.3287530, -0.1361374, 0.3287530, -0.3120778, 0.3120772
6: 0.5845805, 0.8929586, 0.5845805, 0.8929586, -0.1627082, 0.1627066
7: -1.1971323, -0.7447776, -1.1971323, -0.7447776, -0.0953120, 0.0953206
8: -5.1874294, -4.6035657, -5.1874294, -4.6035657, -0.2336428, 0.2337066
9: -5.3642797, -4.8095503, -5.3642797, -4.8095503, -0.2366060, 0.2366540

Time for backsubstitution: 6.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 397
type: DSZ, layer: 1, pos: 3486
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 415
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 3117
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 432
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2798

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 816

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0529793, upper bound: 0.0529840
time: 126.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0529860, upper bound: 0.0529800
time: 14.86 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 147.38 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 147.38
Output dim: 5, lower bound: -0.0530055, upper bound: 0.0530119
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 147.38
Output dim: 5, lower bound: -0.0530053, upper bound: 0.0530098
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 147.38
Output dim: 5, lower bound: -0.0529389, upper bound: 0.0529847
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 147.38
Output dim: 5, lower bound: -0.0529905, upper bound: 0.0529363
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 147.38
Output dim: 5, lower bound: -0.0529903, upper bound: 0.0529947
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 147.38
Output dim: 5, lower bound: -0.0529898, upper bound: 0.0529962
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 147.38
Output dim: 5, lower bound: -0.0529793, upper bound: 0.0529840
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 147.38
Output dim: 5, lower bound: -0.0529860, upper bound: 0.0529800

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7559090, -4.1569338, -4.7559090, -4.1569338, -0.2032351, 0.2031942
1: -4.7411447, -4.2113256, -4.7411447, -4.2113256, -0.1999549, 0.1999253
2: -1.2580136, -1.0103903, -1.2580136, -1.0103903, -0.0831256, 0.0831246
3: -0.0328050, 0.3108093, -0.0328050, 0.3108093, -0.2911204, 0.2911204
4: -0.9833181, -0.6530223, -0.9833181, -0.6530223, -0.1149052, 0.1148997
5: -0.1361374, 0.3287530, -0.1361374, 0.3287530, -0.3121210, 0.3121210
6: 0.5845805, 0.8929586, 0.5845805, 0.8929586, -0.1626750, 0.1626770
7: -1.1971323, -0.7447776, -1.1971323, -0.7447776, -0.0954743, 0.0954675
8: -5.1874294, -4.6035657, -5.1874294, -4.6035657, -0.2346871, 0.2346634
9: -5.3642797, -4.8095503, -5.3642797, -4.8095503, -0.2377023, 0.2377024

Time for backsubstitution: 6.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2599
type: DSZ, layer: 1, pos: 2882
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 445
type: DSZ, layer: 1, pos: 2989
type: DSZ, layer: 1, pos: 3147
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2099
type: DSZ, layer: 1, pos: 3486
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 164
type: DSZ, layer: 1, pos: 3324
type: DSZ, layer: 1, pos: 2830
type: DSZ, layer: 1, pos: 2141
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 3348
type: DSZ, layer: 1, pos: 2797
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2769
type: DSZ, layer: 1, pos: 767
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 2148
type: DSZ, layer: 1, pos: 2212
type: DSZ, layer: 1, pos: 2534
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2897
type: DSZ, layer: 1, pos: 622
type: DSZ, layer: 1, pos: 2763
type: DSZ, layer: 1, pos: 322
type: DSZ, layer: 1, pos: 2810
type: DSZ, layer: 1, pos: 420
type: DSZ, layer: 1, pos: 2877
type: DSZ, layer: 1, pos: 3354
type: DSZ, layer: 1, pos: 3578
type: DSZ, layer: 1, pos: 2858
type: DSZ, layer: 1, pos: 2853
type: DSZ, layer: 1, pos: 2029
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 3322
type: DSZ, layer: 1, pos: 3024
type: DSZ, layer: 1, pos: 2427
type: DSZ, layer: 1, pos: 3366
type: DSZ, layer: 1, pos: 2995
type: DSZ, layer: 1, pos: 2852
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2795
type: DSZ, layer: 1, pos: 3058
type: DSZ, layer: 1, pos: 2825
type: DSZ, layer: 1, pos: 397
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2134
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 2519
type: DSZ, layer: 1, pos: 2140
type: DSZ, layer: 1, pos: 2213
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2572
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 890
type: DSZ, layer: 1, pos: 2748
type: DSZ, layer: 1, pos: 2051
type: DSZ, layer: 1, pos: 3345
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2039
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2215
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2030
type: DSZ, layer: 1, pos: 3534
type: DSZ, layer: 1, pos: 2671
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 382
type: DSZ, layer: 1, pos: 621
type: DSZ, layer: 1, pos: 2383
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 3121
type: DSZ, layer: 1, pos: 2306
type: DSZ, layer: 1, pos: 3531
type: DSZ, layer: 1, pos: 2842
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2312
type: DSZ, layer: 1, pos: 2043
type: DSZ, layer: 1, pos: 3427
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2293
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2040
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 3426
type: DSZ, layer: 1, pos: 3338
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2036
type: DSZ, layer: 1, pos: 566
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 895
type: DSZ, layer: 1, pos: 3556
type: DSZ, layer: 1, pos: 2641
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 2349
type: DSZ, layer: 1, pos: 3562
type: DSZ, layer: 1, pos: 876
type: DSZ, layer: 1, pos: 2867
type: DSZ, layer: 1, pos: 3552
type: DSZ, layer: 1, pos: 2765
type: DSZ, layer: 1, pos: 2319
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 2580
type: DSZ, layer: 1, pos: 2050
type: DSZ, layer: 1, pos: 2771
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 816
type: DSZ, layer: 1, pos: 385
type: DSZ, layer: 1, pos: 3020
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 3559
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2035
type: DSZ, layer: 1, pos: 2907
type: DSZ, layer: 1, pos: 626
type: DSZ, layer: 1, pos: 2798
type: DSZ, layer: 1, pos: 2824
type: DSZ, layer: 1, pos: 2112
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2656
type: DSZ, layer: 1, pos: 432
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 877
type: DSZ, layer: 1, pos: 3326
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 654
type: DSZ, layer: 1, pos: 2395
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2031
type: DSZ, layer: 1, pos: 631
type: DSZ, layer: 1, pos: 2084
type: DSZ, layer: 1, pos: 2037
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2796
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 3331
type: DSZ, layer: 1, pos: 2445
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 415
type: DSZ, layer: 1, pos: 396
type: DSZ, layer: 1, pos: 3339
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 661
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2749
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 9
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 23
type: DSZ, layer: 1, pos: 653
type: DSZ, layer: 1, pos: 762
type: DSZ, layer: 1, pos: 639
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2613

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2599

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0529701, upper bound: 0.0529613
time: 273.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0529587, upper bound: 0.0529773
time: 5.62 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 284.80 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 284.80
Output dim: 5, lower bound: -0.0529701, upper bound: 0.0529613
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 284.80
Output dim: 5, lower bound: -0.0529587, upper bound: 0.0529773
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 284.80
Output dim: 5, lower bound: -0.0530053, upper bound: 0.0530098
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 284.80
Output dim: 5, lower bound: -0.0529389, upper bound: 0.0529847
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 284.80
Output dim: 5, lower bound: -0.0529905, upper bound: 0.0529363
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 284.80
Output dim: 5, lower bound: -0.0529903, upper bound: 0.0529947
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 284.80
Output dim: 5, lower bound: -0.0529898, upper bound: 0.0529962
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 284.80
Output dim: 5, lower bound: -0.0529793, upper bound: 0.0529840
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 284.80
Output dim: 5, lower bound: -0.0529860, upper bound: 0.0529800

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 176.71 + 1688.54 = 1865.25 seconds
