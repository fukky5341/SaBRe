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
execution time: IAR + RelationalAnalysis = 7.80 + 21.72 = 29.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0376758, upper bound: 0.0377053

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2090
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2090

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376690, upper bound: 0.0376797
time: 185.35 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376693, upper bound: 0.0376892
time: 7.69 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 193.12 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 193.12
Output dim: 2, lower bound: -0.0376690, upper bound: 0.0376797
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 193.12
Output dim: 2, lower bound: -0.0376693, upper bound: 0.0376892

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3694036, 1.3692150
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7875760, 1.7873507
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4152347, 0.4152387
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2197132, 0.2197751
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7738217, 0.7738236
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2612557, 0.2613424
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1890797, 0.1890888
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0992806, 0.0992904
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9351754, 0.9350610
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7929094, 0.7927676

Time for backsubstitution: 5.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2089

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376595, upper bound: 0.0376819
time: 9.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376643, upper bound: 0.0376707
time: 166.28 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3692150, 1.3694036
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7873507, 1.7875762
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4152388, 0.4152347
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2197751, 0.2197132
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7738236, 0.7738217
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2613424, 0.2612557
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1890887, 0.1890797
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0992904, 0.0992806
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9350610, 0.9351754
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7927676, 0.7929093

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2089

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376597, upper bound: 0.0376899
time: 5.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376649, upper bound: 0.0376865
time: 4.86 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 16.92 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 16.92
Output dim: 2, lower bound: -0.0376595, upper bound: 0.0376819
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 16.92
Output dim: 2, lower bound: -0.0376643, upper bound: 0.0376707
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 16.92
Output dim: 2, lower bound: -0.0376597, upper bound: 0.0376899
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 16.92
Output dim: 2, lower bound: -0.0376649, upper bound: 0.0376865

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3627678, 1.3623128
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7777324, 1.7771907
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4151751, 0.4151883
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2166318, 0.2168037
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7737393, 0.7737482
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2572564, 0.2574744
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1883666, 0.1883950
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0986409, 0.0986686
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9305382, 0.9302707
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7863710, 0.7860273

Time for backsubstitution: 6.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376548, upper bound: 0.0376727
time: 152.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376548, upper bound: 0.0376752
time: 164.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3625014, 1.3625166
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7774162, 1.7774105
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4151802, 0.4151793
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2167141, 0.2166936
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7737427, 0.7737411
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2573543, 0.2573431
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1883779, 0.1883757
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0986525, 0.0986508
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9303850, 0.9303858
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7861692, 0.7861735

Time for backsubstitution: 6.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376596, upper bound: 0.0376620
time: 123.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376596, upper bound: 0.0376653
time: 119.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3625166, 1.3625014
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7774105, 1.7774162
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4151792, 0.4151802
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2166936, 0.2167141
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7737411, 0.7737427
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2573431, 0.2573543
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1883757, 0.1883779
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0986508, 0.0986525
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9303858, 0.9303851
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7861735, 0.7861692

Time for backsubstitution: 6.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376550, upper bound: 0.0376541
time: 73.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376550, upper bound: 0.0376681
time: 176.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3623128, 1.3627679
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7771912, 1.7777324
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4151883, 0.4151752
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2168036, 0.2166318
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7737482, 0.7737393
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2574744, 0.2572564
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1883950, 0.1883666
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0986686, 0.0986409
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9302707, 0.9305382
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7860274, 0.7863711

Time for backsubstitution: 6.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376602, upper bound: 0.0376619
time: 29.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376602, upper bound: 0.0376674
time: 68.54 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 104.77 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 104.77
Output dim: 2, lower bound: -0.0376548, upper bound: 0.0376727
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 104.77
Output dim: 2, lower bound: -0.0376548, upper bound: 0.0376752
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 104.77
Output dim: 2, lower bound: -0.0376596, upper bound: 0.0376620
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 104.77
Output dim: 2, lower bound: -0.0376596, upper bound: 0.0376653
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 104.77
Output dim: 2, lower bound: -0.0376550, upper bound: 0.0376541
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 104.77
Output dim: 2, lower bound: -0.0376550, upper bound: 0.0376681
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 104.77
Output dim: 2, lower bound: -0.0376602, upper bound: 0.0376619
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 104.77
Output dim: 2, lower bound: -0.0376602, upper bound: 0.0376674

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3629092, 1.3622477
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7778783, 1.7770214
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4151734, 0.4151919
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2165994, 0.2168325
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7737310, 0.7737554
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2572208, 0.2575015
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1883339, 0.1884519
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0986390, 0.0986663
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9306033, 0.9302409
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7864635, 0.7859215

Time for backsubstitution: 6.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2074

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376500, upper bound: 0.0376665
time: 116.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376506, upper bound: 0.0376788
time: 6.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3627026, 1.3623128
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7775629, 1.7771907
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4151751, 0.4151864
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2166318, 0.2167712
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7737393, 0.7737398
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2572564, 0.2574388
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1883666, 0.1883622
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0986409, 0.0986666
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9305084, 0.9302707
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7862651, 0.7860273

Time for backsubstitution: 6.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2074

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376500, upper bound: 0.0376669
time: 136.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376506, upper bound: 0.0376788
time: 6.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3626428, 1.3624516
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7775624, 1.7772412
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4151784, 0.4151828
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2166818, 0.2167225
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7737343, 0.7737484
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2573188, 0.2573701
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1883451, 0.1884325
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0986505, 0.0986485
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9304500, 0.9303560
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7862617, 0.7860676

Time for backsubstitution: 6.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2074

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376548, upper bound: 0.0376809
time: 4.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376554, upper bound: 0.0376766
time: 6.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3624363, 1.3625166
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7772470, 1.7774105
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4151802, 0.4151773
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2167141, 0.2166613
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7737427, 0.7737328
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2573543, 0.2573075
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1883779, 0.1883429
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0986525, 0.0986488
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9303551, 0.9303858
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7860633, 0.7861735

Time for backsubstitution: 6.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2074

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376548, upper bound: 0.0376809
time: 4.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376554, upper bound: 0.0376766
time: 6.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3626634, 1.3624363
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7775636, 1.7772470
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4151775, 0.4151840
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2166613, 0.2167442
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7737328, 0.7737501
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2573075, 0.2573839
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1883429, 0.1884358
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0986488, 0.0986504
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9304531, 0.9303551
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7862718, 0.7860633

Time for backsubstitution: 6.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2074

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376502, upper bound: 0.0376832
time: 5.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376508, upper bound: 0.0376664
time: 82.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3624516, 1.3625014
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7772412, 1.7774162
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4151792, 0.4151783
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2166936, 0.2166818
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7737411, 0.7737343
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2573431, 0.2573188
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1883757, 0.1883451
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0986508, 0.0986505
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9303560, 0.9303851
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7860677, 0.7861692

Time for backsubstitution: 6.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2074

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376502, upper bound: 0.0376841
time: 5.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376508, upper bound: 0.0376655
time: 111.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3624594, 1.3627026
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7773442, 1.7775631
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4151865, 0.4151790
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2167713, 0.2166619
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7737398, 0.7737467
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2574389, 0.2572860
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1883622, 0.1884245
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0986666, 0.0986388
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9303379, 0.9305084
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7861257, 0.7862653

Time for backsubstitution: 6.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2074

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376554, upper bound: 0.0376699
time: 5.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376560, upper bound: 0.0376660
time: 19.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3622477, 1.3627679
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7770216, 1.7777324
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4151883, 0.4151733
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2168036, 0.2165994
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7737482, 0.7737310
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2574744, 0.2572209
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1883950, 0.1883339
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0986686, 0.0986390
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9302408, 0.9305382
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7859216, 0.7863711

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2074
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2074

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376554, upper bound: 0.0376699
time: 5.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376560, upper bound: 0.0376674
time: 22.04 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 33.96 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.96
Output dim: 2, lower bound: -0.0376500, upper bound: 0.0376665
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.96
Output dim: 2, lower bound: -0.0376506, upper bound: 0.0376788
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.96
Output dim: 2, lower bound: -0.0376500, upper bound: 0.0376669
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.96
Output dim: 2, lower bound: -0.0376506, upper bound: 0.0376788
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.96
Output dim: 2, lower bound: -0.0376548, upper bound: 0.0376809
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.96
Output dim: 2, lower bound: -0.0376554, upper bound: 0.0376766
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.96
Output dim: 2, lower bound: -0.0376548, upper bound: 0.0376809
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.96
Output dim: 2, lower bound: -0.0376554, upper bound: 0.0376766
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.96
Output dim: 2, lower bound: -0.0376502, upper bound: 0.0376832
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.96
Output dim: 2, lower bound: -0.0376508, upper bound: 0.0376664
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.96
Output dim: 2, lower bound: -0.0376502, upper bound: 0.0376841
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.96
Output dim: 2, lower bound: -0.0376508, upper bound: 0.0376655
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.96
Output dim: 2, lower bound: -0.0376554, upper bound: 0.0376699
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.96
Output dim: 2, lower bound: -0.0376560, upper bound: 0.0376660
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.96
Output dim: 2, lower bound: -0.0376554, upper bound: 0.0376699
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.96
Output dim: 2, lower bound: -0.0376560, upper bound: 0.0376674

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3629054, 1.3622351
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7778559, 1.7769790
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4151714, 0.4151907
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2165738, 0.2168230
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7737304, 0.7737551
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2571867, 0.2574866
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1883298, 0.1884471
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0986373, 0.0986647
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9305911, 0.9302132
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7864364, 0.7858628

Time for backsubstitution: 6.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2091

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376381, upper bound: 0.0376554
time: 223.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376408, upper bound: 0.0376542
time: 9.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3628966, 1.3622441
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7778358, 1.7769995
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4151723, 0.4151897
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2165906, 0.2168068
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7737306, 0.7737548
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2572066, 0.2574672
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1883292, 0.1884478
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0986374, 0.0986647
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9305756, 0.9302287
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7864047, 0.7858948

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2091

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376390, upper bound: 0.0376695
time: 4.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376417, upper bound: 0.0376683
time: 5.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3626989, 1.3622999
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7775402, 1.7771482
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4151731, 0.4151855
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2166061, 0.2167618
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7737386, 0.7737395
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2572221, 0.2574238
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1883625, 0.1883575
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0986392, 0.0986650
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9304962, 0.9302430
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7862381, 0.7859684

Time for backsubstitution: 6.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2091

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376381, upper bound: 0.0376585
time: 222.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376408, upper bound: 0.0376542
time: 9.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3626900, 1.3623090
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7775207, 1.7771688
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4151741, 0.4151845
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2166229, 0.2167456
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7737389, 0.7737392
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2572420, 0.2574047
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1883619, 0.1883582
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0986393, 0.0986650
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9304807, 0.9302583
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7862063, 0.7860008

Time for backsubstitution: 6.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2091

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376390, upper bound: 0.0376695
time: 4.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376417, upper bound: 0.0376683
time: 5.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3626392, 1.3624389
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7775400, 1.7771988
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4151764, 0.4151816
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2166561, 0.2167135
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7737337, 0.7737479
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2572846, 0.2573553
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1883410, 0.1884278
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0986489, 0.0986469
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9304379, 0.9303284
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7862349, 0.7860087

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2091

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376433, upper bound: 0.0376556
time: 90.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376459, upper bound: 0.0376490
time: 155.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3626301, 1.3624480
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7775202, 1.7772188
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4151773, 0.4151807
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2166727, 0.2166969
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7737340, 0.7737478
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2573040, 0.2573360
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1883404, 0.1884285
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0986489, 0.0986468
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9304223, 0.9303439
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7862028, 0.7860407

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2091

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376439, upper bound: 0.0376693
time: 6.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376466, upper bound: 0.0376630
time: 6.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3624327, 1.3625038
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7772245, 1.7773681
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4151781, 0.4151764
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2166884, 0.2166522
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7737421, 0.7737324
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2573200, 0.2572927
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1883738, 0.1883382
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0986508, 0.0986472
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9303432, 0.9303579
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7860365, 0.7861148

Time for backsubstitution: 6.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2091

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376433, upper bound: 0.0376466
time: 75.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376459, upper bound: 0.0376414
time: 214.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3624237, 1.3625128
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7772045, 1.7773881
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4151791, 0.4151754
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2167051, 0.2166356
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7737422, 0.7737322
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2573394, 0.2572734
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1883731, 0.1883388
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0986509, 0.0986472
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9303275, 0.9303737
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7860044, 0.7861465

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2091

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376439, upper bound: 0.0376693
time: 6.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376466, upper bound: 0.0376630
time: 6.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3626598, 1.3624237
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7775412, 1.7772045
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4151754, 0.4151828
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2166356, 0.2167352
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7737322, 0.7737497
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2572733, 0.2573691
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1883388, 0.1884311
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0986472, 0.0986488
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9304409, 0.9303277
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7862448, 0.7860044

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2091

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376381, upper bound: 0.0376526
time: 58.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376429, upper bound: 0.0376447
time: 208.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3626508, 1.3624327
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7775214, 1.7772245
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4151764, 0.4151819
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2166522, 0.2167185
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7737324, 0.7737495
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2572927, 0.2573498
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1883382, 0.1884317
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0986472, 0.0986487
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9304255, 0.9303432
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7862130, 0.7860366

Time for backsubstitution: 6.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 2091

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376390, upper bound: 0.0376628
time: 65.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376438, upper bound: 0.0376552
time: 130.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -4.7840815, -2.9988453, -4.7840815, -2.9988453, -1.3624480, 1.3624885
1: -4.8798532, -2.3031998, -4.8798532, -2.3031998, -1.7772188, 1.7773738
2: -0.5827625, -0.0913776, -0.5827625, -0.0913776, -0.4151772, 0.4151773
3: -0.5533736, -0.0466069, -0.5533736, -0.0466069, -0.2166679, 0.2166727
4: -0.6760729, 0.1213156, -0.6760729, 0.1213156, -0.7737405, 0.7737340
5: -0.8906057, -0.2062376, -0.8906057, -0.2062376, -0.2573088, 0.2573039
6: -0.2617079, 0.2050382, -0.2617079, 0.2050382, -0.1883716, 0.1883404
7: -0.8957834, -0.4121985, -0.8957834, -0.4121985, -0.0986491, 0.0986489
8: -6.1101456, -4.2800198, -6.1101456, -4.2800198, -0.9303439, 0.9303570
9: -4.2516184, -2.6375861, -4.2516184, -2.6375861, -0.7860407, 0.7861103

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2091
type: DSZ, layer: 1, pos: 2104
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2059
type: DSZ, layer: 1, pos: 2539
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 2509
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2527
type: DSZ, layer: 1, pos: 2542
type: DSZ, layer: 1, pos: 51
type: DSZ, layer: 1, pos: 740
type: DSZ, layer: 1, pos: 739
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 724
type: DSZ, layer: 1, pos: 2048
type: DSZ, layer: 1, pos: 8
type: DSZ, layer: 1, pos: 2062
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 699
type: DSZ, layer: 1, pos: 2033
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 742
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 39
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 756
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2081
type: DSZ, layer: 1, pos: 38
type: DSZ, layer: 1, pos: 684
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 683
type: DSZ, layer: 1, pos: 2066
type: DSZ, layer: 1, pos: 2261
type: DSZ, layer: 1, pos: 335
type: DSZ, layer: 1, pos: 530
type: DSZ, layer: 1, pos: 3
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 5
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 17
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 19
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 31
type: DSZ, layer: 1, pos: 32
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 47
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 61
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 73
type: DSZ, layer: 1, pos: 74
type: DSZ, layer: 1, pos: 77
type: DSZ, layer: 1, pos: 84
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 87
type: DSZ, layer: 1, pos: 88
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 91
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 115
type: DSZ, layer: 1, pos: 116
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 121
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 141
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 152
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 162
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 172
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 184
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 186
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 675
type: DSZ, layer: 1, pos: 676
type: DSZ, layer: 1, pos: 694
type: DSZ, layer: 1, pos: 695
type: DSZ, layer: 1, pos: 703
type: DSZ, layer: 1, pos: 704
type: DSZ, layer: 1, pos: 732
type: DSZ, layer: 1, pos: 733
type: DSZ, layer: 1, pos: 774
type: DSZ, layer: 1, pos: 776
type: DSZ, layer: 1, pos: 777
type: DSZ, layer: 1, pos: 780
type: DSZ, layer: 1, pos: 781
type: DSZ, layer: 1, pos: 782
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 810
type: DSZ, layer: 1, pos: 811
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 853
type: DSZ, layer: 1, pos: 854
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 870
type: DSZ, layer: 1, pos: 872
type: DSZ, layer: 1, pos: 873
type: DSZ, layer: 1, pos: 874
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 882
type: DSZ, layer: 1, pos: 883
type: DSZ, layer: 1, pos: 884
type: DSZ, layer: 1, pos: 2026
type: DSZ, layer: 1, pos: 2032
type: DSZ, layer: 1, pos: 2038
type: DSZ, layer: 1, pos: 2044
type: DSZ, layer: 1, pos: 2045
type: DSZ, layer: 1, pos: 2046
type: DSZ, layer: 1, pos: 2047
type: DSZ, layer: 1, pos: 2052
type: DSZ, layer: 1, pos: 2053
type: DSZ, layer: 1, pos: 2054
type: DSZ, layer: 1, pos: 2058
type: DSZ, layer: 1, pos: 2067
type: DSZ, layer: 1, pos: 2072
type: DSZ, layer: 1, pos: 2073
type: DSZ, layer: 1, pos: 2082
type: DSZ, layer: 1, pos: 2083
type: DSZ, layer: 1, pos: 2087
type: DSZ, layer: 1, pos: 2088
type: DSZ, layer: 1, pos: 2100
type: DSZ, layer: 1, pos: 2101
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2119
type: DSZ, layer: 1, pos: 2120
type: DSZ, layer: 1, pos: 2124
type: DSZ, layer: 1, pos: 2125
type: DSZ, layer: 1, pos: 2126
type: DSZ, layer: 1, pos: 2127
type: DSZ, layer: 1, pos: 2128
type: DSZ, layer: 1, pos: 2129
type: DSZ, layer: 1, pos: 2130
type: DSZ, layer: 1, pos: 2131
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2138
type: DSZ, layer: 1, pos: 2155
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2160
type: DSZ, layer: 1, pos: 2161
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2164
type: DSZ, layer: 1, pos: 2165
type: DSZ, layer: 1, pos: 2175
type: DSZ, layer: 1, pos: 2179
type: DSZ, layer: 1, pos: 2180
type: DSZ, layer: 1, pos: 2181
type: DSZ, layer: 1, pos: 2182
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2194
type: DSZ, layer: 1, pos: 2197
type: DSZ, layer: 1, pos: 2198
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2224
type: DSZ, layer: 1, pos: 2225
type: DSZ, layer: 1, pos: 2226
type: DSZ, layer: 1, pos: 2227
type: DSZ, layer: 1, pos: 2229
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 2234
type: DSZ, layer: 1, pos: 2236
type: DSZ, layer: 1, pos: 2241
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2266
type: DSZ, layer: 1, pos: 2322
type: DSZ, layer: 1, pos: 2341
type: DSZ, layer: 1, pos: 2345
type: DSZ, layer: 1, pos: 2359
type: DSZ, layer: 1, pos: 2360
type: DSZ, layer: 1, pos: 2370
type: DSZ, layer: 1, pos: 2371
type: DSZ, layer: 1, pos: 2376
type: DSZ, layer: 1, pos: 2377
type: DSZ, layer: 1, pos: 2399
type: DSZ, layer: 1, pos: 2411
type: DSZ, layer: 1, pos: 2420
type: DSZ, layer: 1, pos: 2421
type: DSZ, layer: 1, pos: 2425
type: DSZ, layer: 1, pos: 2437
type: DSZ, layer: 1, pos: 2438
type: DSZ, layer: 1, pos: 2444
type: DSZ, layer: 1, pos: 2474
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2482
type: DSZ, layer: 1, pos: 2488
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2503
type: DSZ, layer: 1, pos: 2517
type: DSZ, layer: 1, pos: 2518
type: DSZ, layer: 1, pos: 2532
type: DSZ, layer: 1, pos: 2537
type: DSZ, layer: 1, pos: 2547
type: DSZ, layer: 1, pos: 2552
type: DSZ, layer: 1, pos: 2553
type: DSZ, layer: 1, pos: 2567
type: DSZ, layer: 1, pos: 2576
type: DSZ, layer: 1, pos: 2582
type: DSZ, layer: 1, pos: 2587
type: DSZ, layer: 1, pos: 2590
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 2597
type: DSZ, layer: 1, pos: 2600
type: DSZ, layer: 1, pos: 2602
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2605
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2611
type: DSZ, layer: 1, pos: 2617
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2624
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2632
type: DSZ, layer: 1, pos: 2639
type: DSZ, layer: 1, pos: 2643
type: DSZ, layer: 1, pos: 2647
type: DSZ, layer: 1, pos: 2652
type: DSZ, layer: 1, pos: 2657
type: DSZ, layer: 1, pos: 2658
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2673
type: DSZ, layer: 1, pos: 2677
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2687
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2692
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2696
type: DSZ, layer: 1, pos: 2955
type: DSZ, layer: 1, pos: 2985
type: DSZ, layer: 1, pos: 2986
type: DSZ, layer: 1, pos: 3050
type: DSZ, layer: 1, pos: 3051
type: DSZ, layer: 1, pos: 3077
type: DSZ, layer: 1, pos: 3086
type: DSZ, layer: 1, pos: 3112
type: DSZ, layer: 1, pos: 3115
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3139
type: DSZ, layer: 1, pos: 3144
type: DSZ, layer: 1, pos: 3469

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2091

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0376381, upper bound: 0.0376571
time: 58.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0376429, upper bound: 0.0376467
time: 163.66 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 229.00 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 229.00
Output dim: 2, lower bound: -0.0376381, upper bound: 0.0376554
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 229.00
Output dim: 2, lower bound: -0.0376408, upper bound: 0.0376542
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 229.00
Output dim: 2, lower bound: -0.0376390, upper bound: 0.0376695
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 229.00
Output dim: 2, lower bound: -0.0376417, upper bound: 0.0376683
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 229.00
Output dim: 2, lower bound: -0.0376381, upper bound: 0.0376585
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 229.00
Output dim: 2, lower bound: -0.0376408, upper bound: 0.0376542
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 229.00
Output dim: 2, lower bound: -0.0376390, upper bound: 0.0376695
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 229.00
Output dim: 2, lower bound: -0.0376417, upper bound: 0.0376683
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 229.00
Output dim: 2, lower bound: -0.0376433, upper bound: 0.0376556
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 229.00
Output dim: 2, lower bound: -0.0376459, upper bound: 0.0376490
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 229.00
Output dim: 2, lower bound: -0.0376439, upper bound: 0.0376693
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 229.00
Output dim: 2, lower bound: -0.0376466, upper bound: 0.0376630
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 229.00
Output dim: 2, lower bound: -0.0376433, upper bound: 0.0376466
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 229.00
Output dim: 2, lower bound: -0.0376459, upper bound: 0.0376414
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 229.00
Output dim: 2, lower bound: -0.0376439, upper bound: 0.0376693
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 229.00
Output dim: 2, lower bound: -0.0376466, upper bound: 0.0376630
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 229.00
Output dim: 2, lower bound: -0.0376381, upper bound: 0.0376526
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 229.00
Output dim: 2, lower bound: -0.0376429, upper bound: 0.0376447
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 229.00
Output dim: 2, lower bound: -0.0376390, upper bound: 0.0376628
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 229.00
Output dim: 2, lower bound: -0.0376438, upper bound: 0.0376552
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 229.00
Output dim: 2, lower bound: -0.0376381, upper bound: 0.0376571
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 229.00
Output dim: 2, lower bound: -0.0376429, upper bound: 0.0376467
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 229.00
Output dim: 2, lower bound: -0.0376508, upper bound: 0.0376655
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 229.00
Output dim: 2, lower bound: -0.0376554, upper bound: 0.0376699
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 229.00
Output dim: 2, lower bound: -0.0376560, upper bound: 0.0376660
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 229.00
Output dim: 2, lower bound: -0.0376554, upper bound: 0.0376699
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 229.00
Output dim: 2, lower bound: -0.0376560, upper bound: 0.0376674

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 29.52 + 3725.08 = 3754.60 seconds
