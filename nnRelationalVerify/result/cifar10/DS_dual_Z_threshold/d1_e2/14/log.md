## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 14)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0149927863


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1400471, 0.1400471)
1: (-4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2598646, 0.2598647)
2: (-1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575927, 0.0575927)
3: (-0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550859, 0.1550859)
4: (-1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0418184, 0.0418184)
5: (-0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1615533, 0.1615533)
6: (-1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2302484, 0.2302484)
7: (0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1701828, 0.1701828)
8: (-1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1798248, 0.1798248)
9: (-1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1997391, 0.1997392)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.14 + 18.36 = 25.50 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0150375, upper bound: 0.0150378

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2613

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150373, upper bound: 0.0150315
time: 12.00 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150315, upper bound: 0.0150379
time: 19.26 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 31.33 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 31.33
Output dim: 4, lower bound: -0.0150373, upper bound: 0.0150315
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 31.33
Output dim: 4, lower bound: -0.0150315, upper bound: 0.0150379

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1388429, 0.1389007
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2578523, 0.2578115
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575591, 0.0575618
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550761, 0.1550772
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0418133, 0.0418129
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1615267, 0.1615238
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2302404, 0.2302428
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1700733, 0.1700755
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1783124, 0.1783371
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1980560, 0.1980465

Time for backsubstitution: 5.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 2628

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150206, upper bound: 0.0150066
time: 18.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150070, upper bound: 0.0150136
time: 57.00 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1389007, 0.1388429
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2578116, 0.2578523
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575618, 0.0575591
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550772, 0.1550760
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0418129, 0.0418133
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1615239, 0.1615267
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2302428, 0.2302404
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1700755, 0.1700733
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1783371, 0.1783124
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1980465, 0.1980560

Time for backsubstitution: 5.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2628

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150135, upper bound: 0.0150072
time: 26.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150067, upper bound: 0.0150205
time: 14.34 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 46.22 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 46.22
Output dim: 4, lower bound: -0.0150206, upper bound: 0.0150066
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 46.22
Output dim: 4, lower bound: -0.0150070, upper bound: 0.0150136
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 46.22
Output dim: 4, lower bound: -0.0150135, upper bound: 0.0150072
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 46.22
Output dim: 4, lower bound: -0.0150067, upper bound: 0.0150205

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1354618, 0.1355962
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2493206, 0.2493909
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575667, 0.0575701
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1551529, 0.1551547
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409922, 0.0409762
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1614578, 0.1614531
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301944, 0.2302028
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1687013, 0.1686908
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1695542, 0.1696986
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1890712, 0.1891495

Time for backsubstitution: 5.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2596

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150194, upper bound: 0.0150053
time: 77.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150194, upper bound: 0.0150055
time: 2.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1355068, 0.1355197
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2493554, 0.2492799
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575670, 0.0575694
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1551533, 0.1551540
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409766, 0.0409917
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1614560, 0.1614542
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2302004, 0.2301969
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1686887, 0.1687007
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1696198, 0.1695790
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1891080, 0.1890617

Time for backsubstitution: 5.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2596

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150056, upper bound: 0.0150120
time: 7.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150057, upper bound: 0.0150122
time: 14.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1355197, 0.1355068
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2492799, 0.2493554
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575694, 0.0575670
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1551540, 0.1551533
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409917, 0.0409766
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1614542, 0.1614560
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301969, 0.2302004
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1687007, 0.1686886
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1695790, 0.1696198
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1890617, 0.1891080

Time for backsubstitution: 5.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2596

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150120, upper bound: 0.0150055
time: 9.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150122, upper bound: 0.0150058
time: 21.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1355962, 0.1354618
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2493909, 0.2493206
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575701, 0.0575667
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1551547, 0.1551529
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409762, 0.0409922
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1614531, 0.1614579
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2302028, 0.2301945
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1686908, 0.1687013
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1696985, 0.1695543
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1891496, 0.1890711

Time for backsubstitution: 5.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2596

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150054, upper bound: 0.0150194
time: 12.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150052, upper bound: 0.0150190
time: 58.53 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 76.76 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 76.76
Output dim: 4, lower bound: -0.0150194, upper bound: 0.0150053
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 76.76
Output dim: 4, lower bound: -0.0150194, upper bound: 0.0150055
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 76.76
Output dim: 4, lower bound: -0.0150056, upper bound: 0.0150120
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 76.76
Output dim: 4, lower bound: -0.0150057, upper bound: 0.0150122
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 76.76
Output dim: 4, lower bound: -0.0150120, upper bound: 0.0150055
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 76.76
Output dim: 4, lower bound: -0.0150122, upper bound: 0.0150058
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 76.76
Output dim: 4, lower bound: -0.0150054, upper bound: 0.0150194
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 76.76
Output dim: 4, lower bound: -0.0150052, upper bound: 0.0150190

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1352242, 0.1354103
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2490823, 0.2492136
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575516, 0.0575517
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1551469, 0.1551483
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409834, 0.0409676
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1614523, 0.1614458
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301747, 0.2301753
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1686650, 0.1686399
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1689369, 0.1692586
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1885616, 0.1887658

Time for backsubstitution: 5.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 600

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150163, upper bound: 0.0150054
time: 10.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150191, upper bound: 0.0150028
time: 2.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1352760, 0.1353585
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2491434, 0.2491525
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575484, 0.0575550
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1551465, 0.1551487
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409836, 0.0409674
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1614505, 0.1614475
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301670, 0.2301831
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1686504, 0.1686544
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1691144, 0.1690812
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1886874, 0.1886399

Time for backsubstitution: 5.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 600

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150164, upper bound: 0.0150054
time: 136.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150192, upper bound: 0.0150022
time: 19.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1352692, 0.1353338
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2491171, 0.2491026
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575520, 0.0575510
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1551473, 0.1551476
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409678, 0.0409831
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1614504, 0.1614469
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301807, 0.2301694
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1686523, 0.1686498
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1690025, 0.1691391
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1885984, 0.1886780

Time for backsubstitution: 5.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 600

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150026, upper bound: 0.0150124
time: 2.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150055, upper bound: 0.0150097
time: 2.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1353210, 0.1352820
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2491782, 0.2490415
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575487, 0.0575543
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1551469, 0.1551480
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409680, 0.0409829
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1614487, 0.1614486
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301729, 0.2301772
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1686378, 0.1686643
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1691799, 0.1689617
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1887243, 0.1885521

Time for backsubstitution: 5.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 600

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150027, upper bound: 0.0150116
time: 51.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150055, upper bound: 0.0150096
time: 2.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1352820, 0.1353210
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2490415, 0.2491782
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575543, 0.0575487
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1551481, 0.1551468
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409829, 0.0409680
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1614487, 0.1614487
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301772, 0.2301728
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1686643, 0.1686378
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1689617, 0.1691799
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1885521, 0.1887243

Time for backsubstitution: 5.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 600

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150090, upper bound: 0.0150059
time: 2.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150118, upper bound: 0.0150031
time: 2.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1353338, 0.1352692
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2491026, 0.2491171
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575510, 0.0575519
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1551476, 0.1551473
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409831, 0.0409678
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1614469, 0.1614504
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301695, 0.2301806
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1686498, 0.1686523
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1691391, 0.1690025
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1886780, 0.1885984

Time for backsubstitution: 5.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 600

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150092, upper bound: 0.0150055
time: 2.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150120, upper bound: 0.0150033
time: 2.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1353586, 0.1352760
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2491525, 0.2491434
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575550, 0.0575484
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1551487, 0.1551465
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409674, 0.0409836
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1614475, 0.1614505
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301831, 0.2301670
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1686545, 0.1686504
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1690812, 0.1691144
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1886399, 0.1886874

Time for backsubstitution: 5.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 600

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150024, upper bound: 0.0150192
time: 28.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150051, upper bound: 0.0150169
time: 2.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1354103, 0.1352242
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2492136, 0.2490823
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575517, 0.0575516
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1551483, 0.1551469
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409676, 0.0409834
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1614458, 0.1614523
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301753, 0.2301747
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1686400, 0.1686649
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1692586, 0.1689369
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1887658, 0.1885616

Time for backsubstitution: 5.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 600

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150023, upper bound: 0.0150195
time: 2.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150051, upper bound: 0.0150163
time: 18.34 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 26.63 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 26.63
Output dim: 4, lower bound: -0.0150163, upper bound: 0.0150054
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 26.63
Output dim: 4, lower bound: -0.0150191, upper bound: 0.0150028
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 26.63
Output dim: 4, lower bound: -0.0150164, upper bound: 0.0150054
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 26.63
Output dim: 4, lower bound: -0.0150192, upper bound: 0.0150022
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 26.63
Output dim: 4, lower bound: -0.0150026, upper bound: 0.0150124
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 26.63
Output dim: 4, lower bound: -0.0150055, upper bound: 0.0150097
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 26.63
Output dim: 4, lower bound: -0.0150027, upper bound: 0.0150116
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 26.63
Output dim: 4, lower bound: -0.0150055, upper bound: 0.0150096
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 26.63
Output dim: 4, lower bound: -0.0150090, upper bound: 0.0150059
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 26.63
Output dim: 4, lower bound: -0.0150118, upper bound: 0.0150031
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 26.63
Output dim: 4, lower bound: -0.0150092, upper bound: 0.0150055
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 26.63
Output dim: 4, lower bound: -0.0150120, upper bound: 0.0150033
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 26.63
Output dim: 4, lower bound: -0.0150024, upper bound: 0.0150192
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 26.63
Output dim: 4, lower bound: -0.0150051, upper bound: 0.0150169
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 26.63
Output dim: 4, lower bound: -0.0150023, upper bound: 0.0150195
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 26.63
Output dim: 4, lower bound: -0.0150051, upper bound: 0.0150163

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1351828, 0.1353710
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2489808, 0.2491177
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575018, 0.0575041
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550691, 0.1550673
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409247, 0.0409117
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613441, 0.1613288
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301458, 0.2301465
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1684429, 0.1684263
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1689248, 0.1692477
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1885057, 0.1887121

Time for backsubstitution: 5.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3287

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150155, upper bound: 0.0150005
time: 2.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150114, upper bound: 0.0150045
time: 2.66 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1351849, 0.1353689
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2489864, 0.2491122
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575040, 0.0575019
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550659, 0.1550705
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409275, 0.0409089
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613353, 0.1613376
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301459, 0.2301463
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1684513, 0.1684180
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1689261, 0.1692465
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1885079, 0.1887100

Time for backsubstitution: 5.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3287

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150184, upper bound: 0.0149972
time: 12.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150141, upper bound: 0.0150013
time: 46.18 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1352346, 0.1353192
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2490419, 0.2490566
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0574986, 0.0575074
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550687, 0.1550677
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409249, 0.0409115
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613424, 0.1613306
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301380, 0.2301543
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1684284, 0.1684408
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1691022, 0.1690703
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1886316, 0.1885863

Time for backsubstitution: 5.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3287

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150154, upper bound: 0.0150006
time: 3.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150112, upper bound: 0.0150041
time: 234.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1352367, 0.1353172
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2490475, 0.2490511
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575008, 0.0575052
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550654, 0.1550709
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409277, 0.0409088
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613336, 0.1613394
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301382, 0.2301541
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1684368, 0.1684325
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1691035, 0.1690691
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1886337, 0.1885841

Time for backsubstitution: 5.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 3287

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150182, upper bound: 0.0149975
time: 13.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150141, upper bound: 0.0150017
time: 2.90 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1352278, 0.1352945
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2490157, 0.2490067
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575022, 0.0575035
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550695, 0.1550666
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409092, 0.0409272
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613423, 0.1613300
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301516, 0.2301406
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1684303, 0.1684362
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1689904, 0.1691282
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1885426, 0.1886243

Time for backsubstitution: 5.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3287

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150020, upper bound: 0.0150075
time: 2.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0149977, upper bound: 0.0150110
time: 10.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1352299, 0.1352924
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2490212, 0.2490012
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575044, 0.0575012
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550663, 0.1550698
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409119, 0.0409244
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613334, 0.1613387
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301518, 0.2301404
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1684386, 0.1684278
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1689917, 0.1691269
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1885447, 0.1886221

Time for backsubstitution: 5.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3287

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150047, upper bound: 0.0150045
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150006, upper bound: 0.0150084
time: 2.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1352796, 0.1352427
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2490768, 0.2489456
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0574989, 0.0575067
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550690, 0.1550670
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409093, 0.0409270
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613405, 0.1613317
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301439, 0.2301484
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1684158, 0.1684507
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1691678, 0.1689508
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1886684, 0.1884984

Time for backsubstitution: 5.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3287

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150019, upper bound: 0.0150074
time: 42.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0149977, upper bound: 0.0150116
time: 2.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1352817, 0.1352407
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2490823, 0.2489400
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575011, 0.0575045
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550658, 0.1550702
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409121, 0.0409242
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613317, 0.1613405
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301440, 0.2301482
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1684241, 0.1684423
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1691691, 0.1689495
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1886706, 0.1884963

Time for backsubstitution: 5.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3287

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150048, upper bound: 0.0150043
time: 12.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150006, upper bound: 0.0150087
time: 2.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1352406, 0.1352817
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2489401, 0.2490822
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575045, 0.0575011
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550702, 0.1550658
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409242, 0.0409121
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613405, 0.1613317
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301482, 0.2301440
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1684423, 0.1684241
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1689495, 0.1691691
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1884963, 0.1886706

Time for backsubstitution: 5.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3287

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150082, upper bound: 0.0150007
time: 5.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150041, upper bound: 0.0150048
time: 16.05 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1352427, 0.1352796
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2489456, 0.2490768
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575067, 0.0574989
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550671, 0.1550690
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409270, 0.0409093
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613317, 0.1613405
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301484, 0.2301439
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1684507, 0.1684158
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1689508, 0.1691678
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1884984, 0.1886685

Time for backsubstitution: 5.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3287

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150110, upper bound: 0.0149977
time: 43.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150070, upper bound: 0.0150021
time: 2.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1352924, 0.1352299
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2490012, 0.2490212
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575012, 0.0575044
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550698, 0.1550663
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409244, 0.0409119
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613387, 0.1613334
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301404, 0.2301518
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1684278, 0.1684386
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1691269, 0.1689917
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1886221, 0.1885447

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3287

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150081, upper bound: 0.0150005
time: 16.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150041, upper bound: 0.0150046
time: 14.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1352945, 0.1352278
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2490067, 0.2490156
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575035, 0.0575022
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550666, 0.1550695
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409272, 0.0409092
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613299, 0.1613423
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301407, 0.2301516
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1684361, 0.1684303
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1691282, 0.1689904
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1886243, 0.1885426

Time for backsubstitution: 5.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3287

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150110, upper bound: 0.0149979
time: 35.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150069, upper bound: 0.0150020
time: 14.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1353172, 0.1352367
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2490511, 0.2490474
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575052, 0.0575008
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550709, 0.1550655
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409088, 0.0409277
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613393, 0.1613336
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301541, 0.2301382
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1684325, 0.1684368
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1690691, 0.1691035
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1885841, 0.1886338

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3287

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150016, upper bound: 0.0150140
time: 8.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0149973, upper bound: 0.0150185
time: 12.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1353192, 0.1352346
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2490566, 0.2490419
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575074, 0.0574986
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550677, 0.1550687
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409115, 0.0409249
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613306, 0.1613424
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301542, 0.2301380
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1684408, 0.1684284
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1690704, 0.1691022
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1885862, 0.1886317

Time for backsubstitution: 5.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3287

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150042, upper bound: 0.0150115
time: 9.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150002, upper bound: 0.0150158
time: 3.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1353690, 0.1351849
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2491122, 0.2489863
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575019, 0.0575040
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550705, 0.1550659
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409089, 0.0409275
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613376, 0.1613353
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301464, 0.2301459
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1684180, 0.1684513
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1692465, 0.1689260
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1887100, 0.1885079

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 3287

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150014, upper bound: 0.0150143
time: 25.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0149973, upper bound: 0.0150186
time: 2.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1353710, 0.1351829
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2491177, 0.2489808
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575041, 0.0575018
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550672, 0.1550691
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409117, 0.0409247
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613288, 0.1613441
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301465, 0.2301458
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1684263, 0.1684429
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1692477, 0.1689248
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1887121, 0.1885058

Time for backsubstitution: 5.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3287

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150043, upper bound: 0.0150115
time: 12.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150002, upper bound: 0.0150158
time: 4.18 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 22.22 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150155, upper bound: 0.0150005
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150114, upper bound: 0.0150045
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150184, upper bound: 0.0149972
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150141, upper bound: 0.0150013
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150154, upper bound: 0.0150006
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150112, upper bound: 0.0150041
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150182, upper bound: 0.0149975
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150141, upper bound: 0.0150017
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150020, upper bound: 0.0150075
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0149977, upper bound: 0.0150110
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150047, upper bound: 0.0150045
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150006, upper bound: 0.0150084
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150019, upper bound: 0.0150074
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0149977, upper bound: 0.0150116
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150048, upper bound: 0.0150043
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150006, upper bound: 0.0150087
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150082, upper bound: 0.0150007
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150041, upper bound: 0.0150048
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150110, upper bound: 0.0149977
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150070, upper bound: 0.0150021
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150081, upper bound: 0.0150005
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150041, upper bound: 0.0150046
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150110, upper bound: 0.0149979
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150069, upper bound: 0.0150020
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150016, upper bound: 0.0150140
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0149973, upper bound: 0.0150185
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150042, upper bound: 0.0150115
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150002, upper bound: 0.0150158
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150014, upper bound: 0.0150143
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0149973, upper bound: 0.0150186
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150043, upper bound: 0.0150115
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 22.22
Output dim: 4, lower bound: -0.0150002, upper bound: 0.0150158

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1350864, 0.1352793
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2487071, 0.2488801
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575008, 0.0575026
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550680, 0.1550661
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409045, 0.0408890
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613404, 0.1613253
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301283, 0.2301203
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1684062, 0.1683757
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1687737, 0.1691066
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1882667, 0.1884899

Time for backsubstitution: 5.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150093, upper bound: 0.0149998
time: 16.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150150, upper bound: 0.0149941
time: 2.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1350914, 0.1352746
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2487441, 0.2488439
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575003, 0.0575031
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550680, 0.1550661
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409020, 0.0408915
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613406, 0.1613251
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301195, 0.2301291
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1683923, 0.1683897
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1687841, 0.1690966
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1882841, 0.1884730

Time for backsubstitution: 5.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150047, upper bound: 0.0150042
time: 15.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150109, upper bound: 0.0149990
time: 2.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1350885, 0.1352772
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2487126, 0.2488747
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575030, 0.0575004
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550648, 0.1550693
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409073, 0.0408862
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613315, 0.1613341
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301285, 0.2301201
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1684146, 0.1683673
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1687750, 0.1691054
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1882688, 0.1884878

Time for backsubstitution: 5.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150121, upper bound: 0.0149972
time: 20.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150178, upper bound: 0.0149909
time: 13.11 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1350935, 0.1352725
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2487497, 0.2488384
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575025, 0.0575009
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550648, 0.1550693
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409047, 0.0408887
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613318, 0.1613339
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301197, 0.2301289
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1684006, 0.1683813
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1687854, 0.1690954
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1882862, 0.1884710

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150075, upper bound: 0.0150012
time: 150.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150138, upper bound: 0.0149962
time: 2.79 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1351382, 0.1352274
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2487682, 0.2488190
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0574975, 0.0575059
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550675, 0.1550666
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409047, 0.0408888
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613386, 0.1613270
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301206, 0.2301280
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1683917, 0.1683902
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1689511, 0.1689290
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1883926, 0.1883639

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150094, upper bound: 0.0150000
time: 30.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150150, upper bound: 0.0149942
time: 2.80 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1351433, 0.1352228
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2488053, 0.2487829
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0574971, 0.0575064
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550675, 0.1550666
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409021, 0.0408913
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613389, 0.1613268
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301117, 0.2301368
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1683778, 0.1684042
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1689617, 0.1689193
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1884101, 0.1883472

Time for backsubstitution: 5.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150047, upper bound: 0.0150045
time: 11.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150109, upper bound: 0.0149987
time: 13.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1351402, 0.1352253
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2487737, 0.2488135
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0574998, 0.0575036
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550643, 0.1550698
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409075, 0.0408860
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613299, 0.1613359
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301207, 0.2301279
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1684000, 0.1683818
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1689524, 0.1689278
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1883947, 0.1883618

Time for backsubstitution: 5.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150122, upper bound: 0.0149975
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150178, upper bound: 0.0149913
time: 46.85 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 55.24 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 55.24
Output dim: 4, lower bound: -0.0150093, upper bound: 0.0149998
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 55.24
Output dim: 4, lower bound: -0.0150150, upper bound: 0.0149941
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 55.24
Output dim: 4, lower bound: -0.0150047, upper bound: 0.0150042
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 55.24
Output dim: 4, lower bound: -0.0150109, upper bound: 0.0149990
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 55.24
Output dim: 4, lower bound: -0.0150121, upper bound: 0.0149972
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 55.24
Output dim: 4, lower bound: -0.0150178, upper bound: 0.0149909
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 55.24
Output dim: 4, lower bound: -0.0150075, upper bound: 0.0150012
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 55.24
Output dim: 4, lower bound: -0.0150138, upper bound: 0.0149962
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 55.24
Output dim: 4, lower bound: -0.0150094, upper bound: 0.0150000
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 55.24
Output dim: 4, lower bound: -0.0150150, upper bound: 0.0149942
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 55.24
Output dim: 4, lower bound: -0.0150047, upper bound: 0.0150045
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 55.24
Output dim: 4, lower bound: -0.0150109, upper bound: 0.0149987
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 55.24
Output dim: 4, lower bound: -0.0150122, upper bound: 0.0149975
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 55.24
Output dim: 4, lower bound: -0.0150178, upper bound: 0.0149913
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 55.24
Output dim: 4, lower bound: -0.0150141, upper bound: 0.0150017
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 55.24
Output dim: 4, lower bound: -0.0150020, upper bound: 0.0150075
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 55.24
Output dim: 4, lower bound: -0.0149977, upper bound: 0.0150110
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 55.24
Output dim: 4, lower bound: -0.0150047, upper bound: 0.0150045
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 55.24
Output dim: 4, lower bound: -0.0150006, upper bound: 0.0150084
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 55.24
Output dim: 4, lower bound: -0.0150019, upper bound: 0.0150074
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 55.24
Output dim: 4, lower bound: -0.0149977, upper bound: 0.0150116
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 55.24
Output dim: 4, lower bound: -0.0150048, upper bound: 0.0150043
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 55.24
Output dim: 4, lower bound: -0.0150006, upper bound: 0.0150087
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 55.24
Output dim: 4, lower bound: -0.0150082, upper bound: 0.0150007
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 55.24
Output dim: 4, lower bound: -0.0150041, upper bound: 0.0150048
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 55.24
Output dim: 4, lower bound: -0.0150110, upper bound: 0.0149977
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 55.24
Output dim: 4, lower bound: -0.0150070, upper bound: 0.0150021
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 55.24
Output dim: 4, lower bound: -0.0150081, upper bound: 0.0150005
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 55.24
Output dim: 4, lower bound: -0.0150041, upper bound: 0.0150046
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 55.24
Output dim: 4, lower bound: -0.0150110, upper bound: 0.0149979
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 55.24
Output dim: 4, lower bound: -0.0150069, upper bound: 0.0150020
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 55.24
Output dim: 4, lower bound: -0.0150016, upper bound: 0.0150140
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 55.24
Output dim: 4, lower bound: -0.0149973, upper bound: 0.0150185
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 55.24
Output dim: 4, lower bound: -0.0150042, upper bound: 0.0150115
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 55.24
Output dim: 4, lower bound: -0.0150002, upper bound: 0.0150158
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 55.24
Output dim: 4, lower bound: -0.0150014, upper bound: 0.0150143
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 55.24
Output dim: 4, lower bound: -0.0149973, upper bound: 0.0150186
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 55.24
Output dim: 4, lower bound: -0.0150043, upper bound: 0.0150115
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 55.24
Output dim: 4, lower bound: -0.0150002, upper bound: 0.0150158

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 25.50 + 1802.73 = 1828.23 seconds
