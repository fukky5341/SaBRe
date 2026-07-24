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
execution time: IAR + RelationalAnalysis = 8.31 + 18.02 = 26.33 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0150375, upper bound: 0.0150378

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 771
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 771

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150035, upper bound: 0.0150365
time: 25.81 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150362, upper bound: 0.0150039
time: 2.94 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 28.77 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 28.77
Output dim: 4, lower bound: -0.0150035, upper bound: 0.0150365
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 28.77
Output dim: 4, lower bound: -0.0150362, upper bound: 0.0150039

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1399294, 0.1397449
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2596876, 0.2593751
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575890, 0.0575811
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550721, 0.1550730
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0417701, 0.0417950
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1615418, 0.1615438
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301638, 0.2301715
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1701818, 0.1701817
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1797900, 0.1797233
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1996954, 0.1996143

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2965

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2449

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0148941, upper bound: 0.0150026
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0149693, upper bound: 0.0149273
time: 16.97 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1397449, 0.1399295
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2593750, 0.2596876
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575811, 0.0575890
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1550730, 0.1550721
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0417950, 0.0417701
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1615438, 0.1615418
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301715, 0.2301638
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1701817, 0.1701818
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1797233, 0.1797900
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1996143, 0.1996955

Time for backsubstitution: 6.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 414

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2620

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150350, upper bound: 0.0149561
time: 2.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0149883, upper bound: 0.0150025
time: 17.17 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 26.35 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 26.35
Output dim: 4, lower bound: -0.0148941, upper bound: 0.0150026
DS_DSZ1_DSZ2, status: Status.VERIFIED, split count: 2, time: 26.35
Output dim: 4, lower bound: -0.0149693, upper bound: 0.0149273
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 26.35
Output dim: 4, lower bound: -0.0150350, upper bound: 0.0149561
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 26.35
Output dim: 4, lower bound: -0.0149883, upper bound: 0.0150025

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1392484, 0.1390502
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2559257, 0.2554219
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575615, 0.0575529
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1549035, 0.1548888
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0405997, 0.0406845
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1614619, 0.1614647
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2301537, 0.2301614
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1691732, 0.1692235
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1762375, 0.1759875
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1957428, 0.1954521

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2448

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3502

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0148863, upper bound: 0.0149778
time: 25.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0148697, upper bound: 0.0149949
time: 12.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1328013, 0.1332296
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2508269, 0.2515658
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575896, 0.0576018
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1549815, 0.1549827
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0415938, 0.0415554
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613424, 0.1613332
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2300796, 0.2300732
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1693855, 0.1693755
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1717467, 0.1721598
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1933749, 0.1937829

Time for backsubstitution: 6.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 832

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0149557, upper bound: 0.0149537
time: 23.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150330, upper bound: 0.0148766
time: 5.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1330450, 0.1329859
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2512532, 0.2511395
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575939, 0.0575975
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1549835, 0.1549807
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0415803, 0.0415689
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613351, 0.1613404
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2300810, 0.2300718
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1693754, 0.1693856
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1720932, 0.1718134
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1937018, 0.1934560

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2028

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0149882, upper bound: 0.0149528
time: 17.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0149856, upper bound: 0.0150025
time: 2.71 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 26.42 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 26.42
Output dim: 4, lower bound: -0.0148863, upper bound: 0.0149778
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.42
Output dim: 4, lower bound: -0.0148697, upper bound: 0.0149949
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 26.42
Output dim: 4, lower bound: -0.0149557, upper bound: 0.0149537
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.42
Output dim: 4, lower bound: -0.0150330, upper bound: 0.0148766
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 26.42
Output dim: 4, lower bound: -0.0149882, upper bound: 0.0149528
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 26.42
Output dim: 4, lower bound: -0.0149856, upper bound: 0.0150025

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1391860, 0.1389737
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2552704, 0.2545610
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0574619, 0.0574729
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1544835, 0.1543493
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0404604, 0.0405629
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1614237, 0.1614137
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2288468, 0.2284858
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1680533, 0.1683548
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1758739, 0.1755043
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1952863, 0.1948534

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 3433

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2743

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0148696, upper bound: 0.0149930
time: 17.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0148681, upper bound: 0.0149947
time: 2.72 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1326054, 0.1330646
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2504771, 0.2512631
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575891, 0.0576020
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1549777, 0.1549811
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0414342, 0.0413525
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613328, 0.1613234
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2300778, 0.2300715
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1693198, 0.1693043
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1714111, 0.1718958
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1930757, 0.1935115

Time for backsubstitution: 6.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 804

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150039, upper bound: 0.0148475
time: 14.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150329, upper bound: 0.0148475
time: 2.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1329845, 0.1329346
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2512228, 0.2511086
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575751, 0.0575755
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1549778, 0.1549741
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0415631, 0.0415536
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613344, 0.1613397
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2300488, 0.2300342
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1693647, 0.1693758
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1720086, 0.1717410
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1936949, 0.1934500

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2393

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 773

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0149664, upper bound: 0.0150002
time: 3.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0149834, upper bound: 0.0149833
time: 2.56 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 13.01 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.01
Output dim: 4, lower bound: -0.0148696, upper bound: 0.0149930
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.01
Output dim: 4, lower bound: -0.0148681, upper bound: 0.0149947
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.01
Output dim: 4, lower bound: -0.0150039, upper bound: 0.0148475
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 13.01
Output dim: 4, lower bound: -0.0150329, upper bound: 0.0148475
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 13.01
Output dim: 4, lower bound: -0.0149664, upper bound: 0.0150002
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 13.01
Output dim: 4, lower bound: -0.0149834, upper bound: 0.0149833

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1391263, 0.1389140
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2552317, 0.2545223
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0574460, 0.0574576
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1544761, 0.1543417
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0404555, 0.0405576
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1614226, 0.1614125
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2288442, 0.2284837
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1680491, 0.1683504
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1758021, 0.1754313
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1952424, 0.1948082

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2975

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3294

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0148585, upper bound: 0.0149903
time: 5.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0148670, upper bound: 0.0149821
time: 3.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1391263, 0.1389140
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2552317, 0.2545223
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0574467, 0.0574569
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1544759, 0.1543419
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0404551, 0.0405581
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1614226, 0.1614125
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2288446, 0.2284832
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1680489, 0.1683506
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1758009, 0.1754325
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1952412, 0.1948095

Time for backsubstitution: 6.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2981

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 686

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0148678, upper bound: 0.0149942
time: 86.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0148678, upper bound: 0.0149946
time: 2.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1321988, 0.1326658
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2503933, 0.2511781
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575871, 0.0575998
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1549399, 0.1549374
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409290, 0.0408722
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613173, 0.1613133
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2300370, 0.2300286
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1692289, 0.1692113
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1711209, 0.1716086
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1928489, 0.1932778

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 885

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 414

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0147491, upper bound: 0.0148760
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150035, upper bound: 0.0146216
time: 2.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1322067, 0.1326579
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2503922, 0.2511792
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575869, 0.0575999
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1549340, 0.1549433
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409539, 0.0408473
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613226, 0.1613080
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2300349, 0.2300307
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1692268, 0.1692134
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1711240, 0.1716056
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1928419, 0.1932847

Time for backsubstitution: 6.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2736

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 748

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150331, upper bound: 0.0148452
time: 23.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150308, upper bound: 0.0148473
time: 75.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1313191, 0.1310906
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2487428, 0.2483214
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575242, 0.0575190
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1548996, 0.1548966
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0413652, 0.0413717
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1612552, 0.1612642
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2295049, 0.2294949
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1693604, 0.1693715
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1713240, 0.1709727
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1930755, 0.1927553

Time for backsubstitution: 6.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3287

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2981

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0149630, upper bound: 0.0149933
time: 13.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0149596, upper bound: 0.0149970
time: 6.95 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 26.51 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 26.51
Output dim: 4, lower bound: -0.0148585, upper bound: 0.0149903
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 26.51
Output dim: 4, lower bound: -0.0148670, upper bound: 0.0149821
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 26.51
Output dim: 4, lower bound: -0.0148678, upper bound: 0.0149942
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 26.51
Output dim: 4, lower bound: -0.0148678, upper bound: 0.0149946
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 26.51
Output dim: 4, lower bound: -0.0147491, upper bound: 0.0148760
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 26.51
Output dim: 4, lower bound: -0.0150035, upper bound: 0.0146216
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 26.51
Output dim: 4, lower bound: -0.0150331, upper bound: 0.0148452
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 26.51
Output dim: 4, lower bound: -0.0150308, upper bound: 0.0148473
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 26.51
Output dim: 4, lower bound: -0.0149630, upper bound: 0.0149933
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 26.51
Output dim: 4, lower bound: -0.0149596, upper bound: 0.0149970

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1391248, 0.1389118
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2552297, 0.2545201
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0574372, 0.0574487
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1544756, 0.1543416
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0404542, 0.0405573
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1614210, 0.1614109
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2288423, 0.2284814
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1680441, 0.1683459
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1757921, 0.1754222
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1952332, 0.1948010

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 747

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 156

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0146505, upper bound: 0.0149935
time: 19.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0148669, upper bound: 0.0147772
time: 66.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1391240, 0.1389125
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2552295, 0.2545203
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0574385, 0.0574474
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1544756, 0.1543415
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0404543, 0.0405573
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1614210, 0.1614109
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2288428, 0.2284810
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1680441, 0.1683458
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1757906, 0.1754237
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1952328, 0.1948014

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2596

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 748

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0148676, upper bound: 0.0149924
time: 2.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0148654, upper bound: 0.0149944
time: 3.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1315967, 0.1321711
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2500192, 0.2508954
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575702, 0.0575817
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1546751, 0.1547211
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0406155, 0.0404911
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1611834, 0.1611955
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2297519, 0.2297575
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1692289, 0.1692084
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1710160, 0.1715228
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1928205, 0.1932537

Time for backsubstitution: 6.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3574

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 736

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150035, upper bound: 0.0146129
time: 6.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0149953, upper bound: 0.0146211
time: 19.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1319305, 0.1323667
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2501990, 0.2509759
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575839, 0.0575970
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1549356, 0.1549450
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409137, 0.0408061
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613179, 0.1613035
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2300296, 0.2300255
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1692224, 0.1692089
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1708424, 0.1713092
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1927341, 0.1931709

Time for backsubstitution: 6.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 773

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3069

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0149630, upper bound: 0.0148139
time: 4.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150017, upper bound: 0.0147832
time: 2.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1319155, 0.1323817
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2501889, 0.2509861
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575839, 0.0575969
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1549356, 0.1549450
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0409127, 0.0408071
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613181, 0.1613033
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2300296, 0.2300255
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1692223, 0.1692089
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1708276, 0.1713240
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1927281, 0.1931768

Time for backsubstitution: 6.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2034

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150211, upper bound: 0.0148473
time: 2.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150306, upper bound: 0.0148375
time: 12.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1304431, 0.1301787
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2480277, 0.2475625
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0572324, 0.0572382
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1548895, 0.1548868
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0411868, 0.0411903
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1612453, 0.1612547
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2293862, 0.2293822
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1692966, 0.1693061
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1708353, 0.1704621
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1927016, 0.1923764

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3502

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3278

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0149619, upper bound: 0.0149875
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0149569, upper bound: 0.0149873
time: 315.47 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1304073, 0.1302145
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2479840, 0.2476062
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0572434, 0.0572272
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1548898, 0.1548865
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0411837, 0.0411934
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1612456, 0.1612543
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2293922, 0.2293762
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1692950, 0.1693077
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1708135, 0.1704839
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1926966, 0.1923814

Time for backsubstitution: 6.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 772

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 809

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0149596, upper bound: 0.0149969
time: 11.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0149596, upper bound: 0.0149969
time: 12.45 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 29.99 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 29.99
Output dim: 4, lower bound: -0.0146505, upper bound: 0.0149935
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 29.99
Output dim: 4, lower bound: -0.0148669, upper bound: 0.0147772
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 29.99
Output dim: 4, lower bound: -0.0148676, upper bound: 0.0149924
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 29.99
Output dim: 4, lower bound: -0.0148654, upper bound: 0.0149944
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 29.99
Output dim: 4, lower bound: -0.0150035, upper bound: 0.0146129
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 29.99
Output dim: 4, lower bound: -0.0149953, upper bound: 0.0146211
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 29.99
Output dim: 4, lower bound: -0.0149630, upper bound: 0.0148139
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 29.99
Output dim: 4, lower bound: -0.0150017, upper bound: 0.0147832
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 29.99
Output dim: 4, lower bound: -0.0150211, upper bound: 0.0148473
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 29.99
Output dim: 4, lower bound: -0.0150306, upper bound: 0.0148375
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 29.99
Output dim: 4, lower bound: -0.0149619, upper bound: 0.0149875
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 29.99
Output dim: 4, lower bound: -0.0149569, upper bound: 0.0149873
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 29.99
Output dim: 4, lower bound: -0.0149596, upper bound: 0.0149969
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 29.99
Output dim: 4, lower bound: -0.0149596, upper bound: 0.0149969

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1379461, 0.1376033
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2531297, 0.2522846
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0574240, 0.0574362
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1544790, 0.1543406
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0393903, 0.0396061
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613602, 0.1613495
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2288286, 0.2284678
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1676419, 0.1679649
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1738830, 0.1732871
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1933672, 0.1928475

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2560

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 747

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0146504, upper bound: 0.0149882
time: 19.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0146456, upper bound: 0.0149934
time: 12.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1388328, 0.1386363
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2550263, 0.2543271
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0574355, 0.0574444
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1544772, 0.1543432
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0404131, 0.0405171
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1614164, 0.1614062
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2288376, 0.2284758
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1680396, 0.1683412
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1754943, 0.1751421
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1951190, 0.1946936

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2744

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2458

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0148475, upper bound: 0.0149929
time: 2.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0148641, upper bound: 0.0149758
time: 19.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1313406, 0.1319009
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2499509, 0.2508220
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0574227, 0.0574389
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1546687, 0.1547150
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0405345, 0.0404058
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1611792, 0.1611914
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2297322, 0.2297383
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1692158, 0.1691951
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1704584, 0.1709334
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1926383, 0.1930605

Time for backsubstitution: 6.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2735

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 748

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150032, upper bound: 0.0146110
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150010, upper bound: 0.0146128
time: 52.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1313266, 0.1319149
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2499458, 0.2508271
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0574275, 0.0574342
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1546690, 0.1547146
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0405302, 0.0404100
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1611794, 0.1611913
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2297326, 0.2297378
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1692156, 0.1691953
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1704266, 0.1709651
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1926273, 0.1930715

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3433

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2199

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0148580, upper bound: 0.0145792
time: 3.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0149801, upper bound: 0.0145385
time: 2.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1196733, 0.1206745
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2358829, 0.2372660
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575858, 0.0576289
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1543388, 0.1543387
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0389557, 0.0388130
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1605537, 0.1605244
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2286561, 0.2286321
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1692231, 0.1692094
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1640178, 0.1646565
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1880969, 0.1887292

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2734

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2393

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0149172, upper bound: 0.0147451
time: 2.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0149631, upper bound: 0.0146984
time: 13.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1319073, 0.1323735
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2501066, 0.2508948
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575752, 0.0575890
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1549356, 0.1549449
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0408497, 0.0407497
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613148, 0.1613003
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2300239, 0.2300199
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1692214, 0.1692080
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1708033, 0.1712974
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1926265, 0.1930647

Time for backsubstitution: 6.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2738

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 899

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150211, upper bound: 0.0148471
time: 58.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150211, upper bound: 0.0148467
time: 18.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1319072, 0.1323735
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2500975, 0.2509038
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575759, 0.0575883
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1549356, 0.1549449
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0408553, 0.0407441
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613150, 0.1613001
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2300241, 0.2300197
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1692214, 0.1692080
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1708010, 0.1712997
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1926159, 0.1930752

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 772

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 888

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150306, upper bound: 0.0148377
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150306, upper bound: 0.0148374
time: 14.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1304073, 0.1302145
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2479840, 0.2476062
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0572434, 0.0572272
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1548898, 0.1548865
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0411837, 0.0411934
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1612456, 0.1612543
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2293922, 0.2293762
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1692950, 0.1693077
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1708135, 0.1704839
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1926966, 0.1923814

Time for backsubstitution: 6.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2672

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0149306, upper bound: 0.0149969
time: 3.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0149595, upper bound: 0.0149681
time: 2.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1304073, 0.1302145
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2479840, 0.2476062
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0572434, 0.0572272
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1548898, 0.1548865
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0411837, 0.0411934
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1612456, 0.1612543
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2293922, 0.2293762
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1692950, 0.1693077
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1708135, 0.1704839
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1926966, 0.1923814

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3433

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2447

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0149159, upper bound: 0.0149825
time: 2.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0149450, upper bound: 0.0149534
time: 2.83 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 11.99 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 11.99
Output dim: 4, lower bound: -0.0146504, upper bound: 0.0149882
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 11.99
Output dim: 4, lower bound: -0.0146456, upper bound: 0.0149934
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 11.99
Output dim: 4, lower bound: -0.0148475, upper bound: 0.0149929
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 11.99
Output dim: 4, lower bound: -0.0148641, upper bound: 0.0149758
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 11.99
Output dim: 4, lower bound: -0.0150032, upper bound: 0.0146110
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 11.99
Output dim: 4, lower bound: -0.0150010, upper bound: 0.0146128
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 11.99
Output dim: 4, lower bound: -0.0148580, upper bound: 0.0145792
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 11.99
Output dim: 4, lower bound: -0.0149801, upper bound: 0.0145385
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 11.99
Output dim: 4, lower bound: -0.0149172, upper bound: 0.0147451
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 11.99
Output dim: 4, lower bound: -0.0149631, upper bound: 0.0146984
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 11.99
Output dim: 4, lower bound: -0.0150211, upper bound: 0.0148471
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 11.99
Output dim: 4, lower bound: -0.0150211, upper bound: 0.0148467
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 11.99
Output dim: 4, lower bound: -0.0150306, upper bound: 0.0148377
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 11.99
Output dim: 4, lower bound: -0.0150306, upper bound: 0.0148374
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 11.99
Output dim: 4, lower bound: -0.0149306, upper bound: 0.0149969
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 11.99
Output dim: 4, lower bound: -0.0149595, upper bound: 0.0149681
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 11.99
Output dim: 4, lower bound: -0.0149159, upper bound: 0.0149825
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 11.99
Output dim: 4, lower bound: -0.0149450, upper bound: 0.0149534

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1372728, 0.1369634
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2526055, 0.2517847
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0574188, 0.0574309
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1544721, 0.1543335
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0392974, 0.0395172
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613480, 0.1613368
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2288076, 0.2284467
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1676354, 0.1679586
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1731950, 0.1726321
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1931051, 0.1926001

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3110

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2458

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0146274, upper bound: 0.0149919
time: 2.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0146439, upper bound: 0.0149753
time: 28.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1386982, 0.1384729
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2546891, 0.2539262
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0574326, 0.0574409
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1544336, 0.1542964
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0402904, 0.0404112
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613988, 0.1613913
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2288124, 0.2284547
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1678441, 0.1681749
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1748733, 0.1743858
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1945575, 0.1940261

Time for backsubstitution: 6.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 2620
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3181
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 2637

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 778

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0148475, upper bound: 0.0149927
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0148474, upper bound: 0.0149928
time: 2.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1310644, 0.1316097
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2497577, 0.2506187
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0574197, 0.0574359
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1546704, 0.1547167
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0404943, 0.0403646
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1611746, 0.1611869
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2297269, 0.2297330
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1692113, 0.1691905
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1701767, 0.1706369
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1925305, 0.1929466

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2505

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2637

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0149923, upper bound: 0.0145960
time: 265.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0149884, upper bound: 0.0146006
time: 11.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1310494, 0.1316247
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2497477, 0.2506289
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0574197, 0.0574359
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1546704, 0.1547167
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0404933, 0.0403656
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1611747, 0.1611867
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2297270, 0.2297330
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1692113, 0.1691905
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1701619, 0.1706518
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1925246, 0.1929526

Time for backsubstitution: 6.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 899
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2034
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 156
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2458

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3278

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0149998, upper bound: 0.0146069
time: 2.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0149951, upper bound: 0.0146115
time: 42.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.9126606, -2.4069767, -2.9126606, -2.4069767, -0.1319073, 0.1323735
1: -4.6355419, -3.8413548, -4.6355419, -3.8413548, -0.2501066, 0.2508948
2: -1.6610268, -1.4953272, -1.6610268, -1.4953272, -0.0575752, 0.0575890
3: -0.3806537, -0.0913687, -0.3806537, -0.0913687, -0.1549356, 0.1549449
4: -1.1096642, -0.8394910, -1.1096642, -0.8394910, -0.0408497, 0.0407497
5: -0.4606394, -0.2307034, -0.4606394, -0.2307034, -0.1613148, 0.1613003
6: -1.5973693, -1.0873759, -1.5973693, -1.0873759, -0.2300239, 0.2300199
7: 0.0764579, 0.6610365, 0.0764579, 0.6610365, -0.1692214, 0.1692080
8: -1.3467937, -0.6553130, -1.3467937, -0.6553130, -0.1708033, 0.1712974
9: -1.7710223, -1.0460186, -1.7710223, -1.0460186, -0.1926265, 0.1930647

Time for backsubstitution: 6.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2628
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2637
type: DSZ, layer: 1, pos: 2275
type: DSZ, layer: 1, pos: 793
type: DSZ, layer: 1, pos: 2596
type: DSZ, layer: 1, pos: 2540
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 2447
type: DSZ, layer: 1, pos: 2393
type: DSZ, layer: 1, pos: 2736
type: DSZ, layer: 1, pos: 2435
type: DSZ, layer: 1, pos: 2744
type: DSZ, layer: 1, pos: 809
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 2147
type: DSZ, layer: 1, pos: 414
type: DSZ, layer: 1, pos: 2505
type: DSZ, layer: 1, pos: 3106
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2735
type: DSZ, layer: 1, pos: 761
type: DSZ, layer: 1, pos: 2737
type: DSZ, layer: 1, pos: 773
type: DSZ, layer: 1, pos: 3138
type: DSZ, layer: 1, pos: 3502
type: DSZ, layer: 1, pos: 2981
type: DSZ, layer: 1, pos: 3433
type: DSZ, layer: 1, pos: 3294
type: DSZ, layer: 1, pos: 2458
type: DSZ, layer: 1, pos: 778
type: DSZ, layer: 1, pos: 804
type: DSZ, layer: 1, pos: 772
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3022
type: DSZ, layer: 1, pos: 3110
type: DSZ, layer: 1, pos: 2975
type: DSZ, layer: 1, pos: 2672
type: DSZ, layer: 1, pos: 2972
type: DSZ, layer: 1, pos: 747
type: DSZ, layer: 1, pos: 2028
type: DSZ, layer: 1, pos: 2454
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2449
type: DSZ, layer: 1, pos: 267
type: DSZ, layer: 1, pos: 3287
type: DSZ, layer: 1, pos: 2734
type: DSZ, layer: 1, pos: 96
type: DSZ, layer: 1, pos: 3039
type: DSZ, layer: 1, pos: 2419
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 2743
type: DSZ, layer: 1, pos: 3069
type: DSZ, layer: 1, pos: 2428
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 2965
type: DSZ, layer: 1, pos: 3574
type: DSZ, layer: 1, pos: 2512
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 736
type: DSZ, layer: 1, pos: 2560
type: DSZ, layer: 1, pos: 2613
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 59
type: DSZ, layer: 1, pos: 600
type: DSZ, layer: 1, pos: 2448
type: DSZ, layer: 1, pos: 2738
type: DSZ, layer: 1, pos: 3136
type: DSZ, layer: 1, pos: 2555
type: DSZ, layer: 1, pos: 885
type: DSZ, layer: 1, pos: 3133
type: DSZ, layer: 1, pos: 3137
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2554
type: DSZ, layer: 1, pos: 897
type: DSZ, layer: 1, pos: 156

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2628

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150043, upper bound: 0.0148169
time: 5.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0149906, upper bound: 0.0148305
time: 230.44 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 242.37 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 242.37
Output dim: 4, lower bound: -0.0146274, upper bound: 0.0149919
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 242.37
Output dim: 4, lower bound: -0.0146439, upper bound: 0.0149753
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 242.37
Output dim: 4, lower bound: -0.0148475, upper bound: 0.0149927
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 242.37
Output dim: 4, lower bound: -0.0148474, upper bound: 0.0149928
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 242.37
Output dim: 4, lower bound: -0.0149923, upper bound: 0.0145960
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 242.37
Output dim: 4, lower bound: -0.0149884, upper bound: 0.0146006
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 242.37
Output dim: 4, lower bound: -0.0149998, upper bound: 0.0146069
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 242.37
Output dim: 4, lower bound: -0.0149951, upper bound: 0.0146115
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 242.37
Output dim: 4, lower bound: -0.0150043, upper bound: 0.0148169
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 242.37
Output dim: 4, lower bound: -0.0149906, upper bound: 0.0148305
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 242.37
Output dim: 4, lower bound: -0.0150211, upper bound: 0.0148467
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 242.37
Output dim: 4, lower bound: -0.0150306, upper bound: 0.0148377
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 242.37
Output dim: 4, lower bound: -0.0150306, upper bound: 0.0148374
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 242.37
Output dim: 4, lower bound: -0.0149306, upper bound: 0.0149969

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 26.33 + 1954.71 = 1981.04 seconds
