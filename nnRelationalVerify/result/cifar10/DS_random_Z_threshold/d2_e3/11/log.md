## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 11)
Time budget: 3600 seconds
Split limit: 100
Threshold: 0.0376491132


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3763310, 1.3763309)
1: (-4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7966783, 1.7966783)
2: (-0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4152753, 0.4152753)
3: (-0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2225757, 0.2225757)
4: (-0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7738838, 0.7738838)
5: (-0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2647811, 0.2647811)
6: (-0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1898307, 0.1898307)
7: (-0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0998714, 0.0998714)
8: (-6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9393454, 0.9393452)
9: (-4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7985216, 0.7985218)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.32 + 22.24 = 29.56 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0376758, upper bound: 0.0377053

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 839

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2046

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376463, upper bound: 0.0376933
time: 9.76 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376738, upper bound: 0.0376667
time: 6.03 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 15.81 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 15.81
Output dim: 2, lower bound: -0.0376463, upper bound: 0.0376933
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 15.81
Output dim: 2, lower bound: -0.0376738, upper bound: 0.0376667

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3759229, 1.3759332
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7957036, 1.7956796
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4149711, 0.4149794
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2225825, 0.2225825
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7736864, 0.7736909
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2647780, 0.2647766
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1895646, 0.1895711
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0997983, 0.0997968
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9392085, 0.9392009
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7968682, 0.7968242

Time for backsubstitution: 5.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2179

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376378, upper bound: 0.0376638
time: 5.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376384, upper bound: 0.0376813
time: 58.01 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3759332, 1.3759229
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7956793, 1.7957034
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4149794, 0.4149711
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2225825, 0.2225825
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7736909, 0.7736864
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2647767, 0.2647780
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1895711, 0.1895646
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0997968, 0.0997983
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9392009, 0.9392085
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7968242, 0.7968683

Time for backsubstitution: 5.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2474

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 145

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376484, upper bound: 0.0376706
time: 8.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376732, upper bound: 0.0376299
time: 93.46 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 107.29 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 107.29
Output dim: 2, lower bound: -0.0376378, upper bound: 0.0376638
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 107.29
Output dim: 2, lower bound: -0.0376384, upper bound: 0.0376813
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 107.29
Output dim: 2, lower bound: -0.0376484, upper bound: 0.0376706
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 107.29
Output dim: 2, lower bound: -0.0376732, upper bound: 0.0376299

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3719816, 1.3717294
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7866626, 1.7859912
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4147636, 0.4147717
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2217874, 0.2218403
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7734921, 0.7734966
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2634789, 0.2635772
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1895387, 0.1895453
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0991838, 0.0992036
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9294052, 0.9287329
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7856202, 0.7847605

Time for backsubstitution: 5.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 335

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376384, upper bound: 0.0376807
time: 72.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376384, upper bound: 0.0376743
time: 96.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3717190, 1.3719919
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7860153, 1.7866378
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4147633, 0.4147719
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2218403, 0.2217874
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7734920, 0.7734967
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2635785, 0.2634777
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1895388, 0.1895452
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0992051, 0.0991823
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9287405, 0.9293978
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7848046, 0.7855761

Time for backsubstitution: 5.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 172

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3469

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0375721, upper bound: 0.0376730
time: 83.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376372, upper bound: 0.0376032
time: 203.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3757575, 1.3751848
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7954133, 1.7947459
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4149293, 0.4149558
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2224554, 0.2225514
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7736709, 0.7736853
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2645756, 0.2647245
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1895641, 0.1895579
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0996672, 0.0997629
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9389011, 0.9381330
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7965912, 0.7960317

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2083

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376398, upper bound: 0.0376576
time: 138.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376495, upper bound: 0.0376439
time: 58.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3751951, 1.3757472
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7947216, 1.7954373
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4149641, 0.4149210
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2225514, 0.2224554
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7736900, 0.7736663
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2647232, 0.2645769
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1895644, 0.1895576
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0997614, 0.0996687
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9381254, 0.9389086
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7959878, 0.7966354

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2129

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2184

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376651, upper bound: 0.0376376
time: 5.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376642, upper bound: 0.0376250
time: 9.61 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 21.24 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 21.24
Output dim: 2, lower bound: -0.0376384, upper bound: 0.0376807
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 21.24
Output dim: 2, lower bound: -0.0376384, upper bound: 0.0376743
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 21.24
Output dim: 2, lower bound: -0.0375721, upper bound: 0.0376730
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 21.24
Output dim: 2, lower bound: -0.0376372, upper bound: 0.0376032
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 21.24
Output dim: 2, lower bound: -0.0376398, upper bound: 0.0376576
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 21.24
Output dim: 2, lower bound: -0.0376495, upper bound: 0.0376439
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 21.24
Output dim: 2, lower bound: -0.0376651, upper bound: 0.0376376
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 21.24
Output dim: 2, lower bound: -0.0376642, upper bound: 0.0376250

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3719816, 1.3717294
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7866626, 1.7859912
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4147636, 0.4147717
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2217874, 0.2218403
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7734921, 0.7734966
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2634789, 0.2635772
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1895387, 0.1895453
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0991838, 0.0992036
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9294052, 0.9287329
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7856202, 0.7847605

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2033

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376200, upper bound: 0.0376873
time: 7.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376352, upper bound: 0.0376669
time: 8.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3719816, 1.3717294
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7866626, 1.7859912
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4147636, 0.4147717
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2217874, 0.2218403
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7734921, 0.7734966
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2634789, 0.2635772
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1895387, 0.1895453
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0991838, 0.0992036
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9294052, 0.9287329
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7856202, 0.7847605

Time for backsubstitution: 5.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2236

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2266

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376296, upper bound: 0.0376640
time: 197.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376315, upper bound: 0.0376656
time: 115.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3716402, 1.3719320
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7858927, 1.7865119
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4146468, 0.4146637
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2214565, 0.2213897
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7734921, 0.7735157
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2629162, 0.2627994
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1887550, 0.1888265
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0989364, 0.0989071
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9275920, 0.9283364
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7841094, 0.7848943

Time for backsubstitution: 5.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 676

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2399

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0375670, upper bound: 0.0376851
time: 5.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0375670, upper bound: 0.0376812
time: 11.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3757319, 1.3751547
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7951999, 1.7945137
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4149147, 0.4149436
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2223774, 0.2224800
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7736537, 0.7736698
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2644904, 0.2646447
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1895139, 0.1895120
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0996581, 0.0997531
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9388469, 0.9380689
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7963065, 0.7957220

Time for backsubstitution: 6.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376380, upper bound: 0.0376671
time: 8.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376377, upper bound: 0.0376368
time: 97.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3757275, 1.3751593
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7951808, 1.7945328
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4149172, 0.4149410
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2223841, 0.2224734
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7736554, 0.7736682
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2644958, 0.2646393
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1895182, 0.1895077
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0996574, 0.0997538
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9388368, 0.9380789
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7962812, 0.7957473

Time for backsubstitution: 6.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 782

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2474

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376480, upper bound: 0.0376627
time: 6.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376480, upper bound: 0.0376598
time: 11.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3746166, 1.3749278
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7940097, 1.7942195
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4149461, 0.4149026
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2224681, 0.2224033
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7736636, 0.7736410
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2645613, 0.2644615
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1895529, 0.1895485
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0996640, 0.0995712
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9372109, 0.9375918
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7951248, 0.7952476

Time for backsubstitution: 6.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 810

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2345

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376448, upper bound: 0.0376039
time: 8.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376440, upper bound: 0.0375995
time: 152.68 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3743757, 1.3751733
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7935038, 1.7947302
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4149458, 0.4149028
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2224995, 0.2223721
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7736647, 0.7736399
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2646083, 0.2644149
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1895553, 0.1895460
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0996646, 0.0995713
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9368083, 0.9380004
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7946000, 0.7957764

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 2104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2345

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376438, upper bound: 0.0376027
time: 24.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376430, upper bound: 0.0375976
time: 152.00 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 182.39 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 182.39
Output dim: 2, lower bound: -0.0376200, upper bound: 0.0376873
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 182.39
Output dim: 2, lower bound: -0.0376352, upper bound: 0.0376669
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 182.39
Output dim: 2, lower bound: -0.0376296, upper bound: 0.0376640
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 182.39
Output dim: 2, lower bound: -0.0376315, upper bound: 0.0376656
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 182.39
Output dim: 2, lower bound: -0.0375670, upper bound: 0.0376851
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 182.39
Output dim: 2, lower bound: -0.0375670, upper bound: 0.0376812
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 182.39
Output dim: 2, lower bound: -0.0376380, upper bound: 0.0376671
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 182.39
Output dim: 2, lower bound: -0.0376377, upper bound: 0.0376368
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 182.39
Output dim: 2, lower bound: -0.0376480, upper bound: 0.0376627
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 182.39
Output dim: 2, lower bound: -0.0376480, upper bound: 0.0376598
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 182.39
Output dim: 2, lower bound: -0.0376448, upper bound: 0.0376039
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 182.39
Output dim: 2, lower bound: -0.0376440, upper bound: 0.0375995
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 182.39
Output dim: 2, lower bound: -0.0376438, upper bound: 0.0376027
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 182.39
Output dim: 2, lower bound: -0.0376430, upper bound: 0.0375976

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3718674, 1.3716147
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7865543, 1.7858629
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4146813, 0.4147007
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2217802, 0.2218331
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7734448, 0.7734488
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2634704, 0.2635674
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1894754, 0.1894907
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0991621, 0.0991788
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9293618, 0.9286835
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7853947, 0.7844901

Time for backsubstitution: 5.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2083

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0375664, upper bound: 0.0376768
time: 23.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376203, upper bound: 0.0376282
time: 8.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3718669, 1.3716152
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7865343, 1.7858829
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4146926, 0.4146894
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2217802, 0.2218331
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7734443, 0.7734492
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2634692, 0.2635687
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1894841, 0.1894820
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0991590, 0.0991819
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9293561, 0.9286897
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7853497, 0.7845352

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376026, upper bound: 0.0376700
time: 9.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376355, upper bound: 0.0376293
time: 5.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3719959, 1.3717139
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7866607, 1.7859888
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4147630, 0.4147710
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2217862, 0.2218392
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7734914, 0.7734959
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2634776, 0.2635760
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1895375, 0.1895442
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0991838, 0.0992034
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9293967, 0.9287055
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7856176, 0.7847569

Time for backsubstitution: 5.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 91

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376126, upper bound: 0.0376823
time: 4.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376303, upper bound: 0.0376515
time: 114.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3719662, 1.3717437
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7866602, 1.7859893
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4147628, 0.4147712
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2217863, 0.2218391
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7734914, 0.7734959
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2634777, 0.2635759
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1895376, 0.1895441
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0991837, 0.0992036
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9293777, 0.9287245
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7856165, 0.7847579

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2498

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 133

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376315, upper bound: 0.0376643
time: 138.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376320, upper bound: 0.0376818
time: 7.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3716402, 1.3719320
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7858927, 1.7865119
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4146468, 0.4146637
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2214565, 0.2213897
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7734921, 0.7735157
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2629162, 0.2627994
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1887550, 0.1888265
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0989364, 0.0989071
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9275920, 0.9283364
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7841094, 0.7848943

Time for backsubstitution: 6.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 142

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0375593, upper bound: 0.0376651
time: 60.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0375593, upper bound: 0.0376615
time: 135.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3716402, 1.3719320
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7858927, 1.7865119
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4146468, 0.4146637
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2214565, 0.2213897
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7734921, 0.7735157
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2629162, 0.2627994
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1887550, 0.1888265
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0989364, 0.0989071
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9275920, 0.9283364
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7841094, 0.7848943

Time for backsubstitution: 6.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 684

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2180

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0375593, upper bound: 0.0376619
time: 9.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0375607, upper bound: 0.0376653
time: 10.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3754833, 1.3747091
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7946587, 1.7935734
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4149136, 0.4149425
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2223243, 0.2224501
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7736384, 0.7736545
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2644042, 0.2645973
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1895104, 0.1895096
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0996085, 0.0997079
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9381926, 0.9369206
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7956066, 0.7944939

Time for backsubstitution: 6.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 31

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2162

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376081, upper bound: 0.0376386
time: 247.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376261, upper bound: 0.0376368
time: 5.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3757275, 1.3751593
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7951808, 1.7945328
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4149172, 0.4149410
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2223841, 0.2224734
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7736554, 0.7736682
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2644958, 0.2646393
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1895182, 0.1895077
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0996574, 0.0997538
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9388368, 0.9380789
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7962812, 0.7957473

Time for backsubstitution: 6.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 60

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376139, upper bound: 0.0376653
time: 5.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376487, upper bound: 0.0376058
time: 260.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3757275, 1.3751593
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7951808, 1.7945328
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4149172, 0.4149410
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2223841, 0.2224734
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7736554, 0.7736682
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2644958, 0.2646393
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1895182, 0.1895077
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0996574, 0.0997538
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9388368, 0.9380789
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7962812, 0.7957473

Time for backsubstitution: 6.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2184

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2359

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376116, upper bound: 0.0375799
time: 329.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376314, upper bound: 0.0376187
time: 6.27 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 341.88 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 341.88
Output dim: 2, lower bound: -0.0375664, upper bound: 0.0376768
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 341.88
Output dim: 2, lower bound: -0.0376203, upper bound: 0.0376282
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 341.88
Output dim: 2, lower bound: -0.0376026, upper bound: 0.0376700
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 341.88
Output dim: 2, lower bound: -0.0376355, upper bound: 0.0376293
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 341.88
Output dim: 2, lower bound: -0.0376126, upper bound: 0.0376823
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 341.88
Output dim: 2, lower bound: -0.0376303, upper bound: 0.0376515
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 341.88
Output dim: 2, lower bound: -0.0376315, upper bound: 0.0376643
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 341.88
Output dim: 2, lower bound: -0.0376320, upper bound: 0.0376818
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 341.88
Output dim: 2, lower bound: -0.0375593, upper bound: 0.0376651
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 341.88
Output dim: 2, lower bound: -0.0375593, upper bound: 0.0376615
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 341.88
Output dim: 2, lower bound: -0.0375593, upper bound: 0.0376619
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 341.88
Output dim: 2, lower bound: -0.0375607, upper bound: 0.0376653
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 341.88
Output dim: 2, lower bound: -0.0376081, upper bound: 0.0376386
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 341.88
Output dim: 2, lower bound: -0.0376261, upper bound: 0.0376368
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 341.88
Output dim: 2, lower bound: -0.0376139, upper bound: 0.0376653
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 341.88
Output dim: 2, lower bound: -0.0376487, upper bound: 0.0376058
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 341.88
Output dim: 2, lower bound: -0.0376116, upper bound: 0.0375799
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 341.88
Output dim: 2, lower bound: -0.0376314, upper bound: 0.0376187

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3716686, 1.3714018
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7861056, 1.7853746
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4140829, 0.4141443
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2216160, 0.2216805
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7732270, 0.7732456
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2631057, 0.2632277
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1889456, 0.1889977
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0986625, 0.0986417
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9284158, 0.9276667
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7839736, 0.7829599

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0375661, upper bound: 0.0376707
time: 48.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0375625, upper bound: 0.0376760
time: 14.31 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3718162, 1.3715646
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7858295, 1.7851458
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4140509, 0.4140691
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2217768, 0.2218306
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7731254, 0.7731407
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2634616, 0.2635621
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1888342, 0.1888544
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0986305, 0.0986353
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9286953, 0.9280059
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7835554, 0.7826751

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 870

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376024, upper bound: 0.0376520
time: 142.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376014, upper bound: 0.0376681
time: 8.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3719101, 1.3715518
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7866094, 1.7858920
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4147627, 0.4147705
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2217786, 0.2218349
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7734892, 0.7734963
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2634628, 0.2635680
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1895244, 0.1895349
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0991694, 0.0991960
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9292077, 0.9283321
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7855203, 0.7845690

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 854

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376109, upper bound: 0.0376770
time: 62.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376109, upper bound: 0.0376597
time: 47.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3718336, 1.3716280
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7865641, 1.7859373
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4147626, 0.4147707
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2217819, 0.2218316
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7734919, 0.7734938
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2634696, 0.2635612
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1895281, 0.1895311
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0991764, 0.0991890
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9290235, 0.9285164
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7854298, 0.7846596

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 3469
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2597

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2628

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376105, upper bound: 0.0376317
time: 189.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376105, upper bound: 0.0376020
time: 181.06 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 376.45 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 376.45
Output dim: 2, lower bound: -0.0375661, upper bound: 0.0376707
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 376.45
Output dim: 2, lower bound: -0.0375625, upper bound: 0.0376760
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 376.45
Output dim: 2, lower bound: -0.0376024, upper bound: 0.0376520
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 376.45
Output dim: 2, lower bound: -0.0376014, upper bound: 0.0376681
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 376.45
Output dim: 2, lower bound: -0.0376109, upper bound: 0.0376770
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 376.45
Output dim: 2, lower bound: -0.0376109, upper bound: 0.0376597
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 376.45
Output dim: 2, lower bound: -0.0376105, upper bound: 0.0376317
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 376.45
Output dim: 2, lower bound: -0.0376105, upper bound: 0.0376020
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 376.45
Output dim: 2, lower bound: -0.0376315, upper bound: 0.0376643
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 376.45
Output dim: 2, lower bound: -0.0376320, upper bound: 0.0376818
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 376.45
Output dim: 2, lower bound: -0.0375593, upper bound: 0.0376651
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 376.45
Output dim: 2, lower bound: -0.0375593, upper bound: 0.0376615
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 376.45
Output dim: 2, lower bound: -0.0375593, upper bound: 0.0376619
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 376.45
Output dim: 2, lower bound: -0.0375607, upper bound: 0.0376653
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 376.45
Output dim: 2, lower bound: -0.0376139, upper bound: 0.0376653

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 29.56 + 3890.17 = 3919.73 seconds
