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
execution time: IAR + RelationalAnalysis = 7.17 + 18.91 = 26.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0150375, upper bound: 0.0150378

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2454
type: A, layer: 1, pos: 2435
type: A, layer: 1, pos: 2449
type: A, layer: 1, pos: 2393
type: A, layer: 1, pos: 156
type: A, layer: 1, pos: 2419
type: A, layer: 1, pos: 3110
type: A, layer: 1, pos: 3069
type: A, layer: 1, pos: 3039
type: A, layer: 1, pos: 2447
type: A, layer: 1, pos: 414
type: A, layer: 1, pos: 2448
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 3181
type: A, layer: 1, pos: 3502
type: A, layer: 1, pos: 773
type: A, layer: 1, pos: 3133
type: A, layer: 1, pos: 804
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 2458
type: A, layer: 1, pos: 2628
type: A, layer: 1, pos: 772
type: A, layer: 1, pos: 2620
type: A, layer: 1, pos: 2512
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2738
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 2972
type: A, layer: 1, pos: 771
type: A, layer: 1, pos: 2428
type: A, layer: 1, pos: 2981
type: A, layer: 1, pos: 2737
type: A, layer: 1, pos: 2034
type: A, layer: 1, pos: 2736
type: A, layer: 1, pos: 2540
type: A, layer: 1, pos: 2554
type: A, layer: 1, pos: 2672
type: A, layer: 1, pos: 2505
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 752
type: A, layer: 1, pos: 3433
type: A, layer: 1, pos: 3106
type: A, layer: 1, pos: 2732
type: A, layer: 1, pos: 2637
type: A, layer: 1, pos: 3294
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 2028
type: A, layer: 1, pos: 267
type: A, layer: 1, pos: 747
type: A, layer: 1, pos: 2596
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2555
type: A, layer: 1, pos: 3022
type: A, layer: 1, pos: 2965
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 2735
type: A, layer: 1, pos: 600
type: A, layer: 1, pos: 761
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 736
type: A, layer: 1, pos: 2743
type: A, layer: 1, pos: 2734
type: A, layer: 1, pos: 2613
type: A, layer: 1, pos: 686
type: A, layer: 1, pos: 3278
type: A, layer: 1, pos: 2147
type: A, layer: 1, pos: 2560
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 3574
type: A, layer: 1, pos: 778
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 809
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 897
type: A, layer: 1, pos: 899
type: A, layer: 1, pos: 2744
type: A, layer: 1, pos: 3136
type: A, layer: 1, pos: 3137
type: A, layer: 1, pos: 3138

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 2454

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0148935, upper bound: 0.0150102
time: 37.46 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0150103, upper bound: 0.0150105
time: 103.02 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 140.54 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 140.54
Output dim: 4, lower bound: -0.0148935, upper bound: 0.0150102
NS_A2, status: Status.UNKNOWN, split count: 1, time: 140.54
Output dim: 4, lower bound: -0.0150103, upper bound: 0.0150105

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -2.9126599, -2.4077656, -2.9126601, -2.4076247, -0.1392018, 0.1390145
1: -4.6355419, -3.8444030, -4.6355419, -3.8438559, -0.2564818, 0.2557286
2: -1.6610159, -1.4953274, -1.6610177, -1.4953272, -0.0575344, 0.0575367
3: -0.3805846, -0.0914032, -0.3805967, -0.0913972, -0.1547366, 0.1547412
4: -1.1082958, -0.8394912, -1.1085420, -0.8394912, -0.0400454, 0.0403688
5: -0.4603674, -0.2307034, -0.4604160, -0.2307034, -0.1612718, 0.1613211
6: -1.5970846, -1.0873761, -1.5971334, -1.0873759, -0.2299543, 0.2300056
7: 0.0769322, 0.6610362, 0.0768488, 0.6610363, -0.1695365, 0.1696542
8: -1.3467934, -0.6578870, -1.3467934, -0.6574287, -0.1771673, 0.1765734
9: -1.7710223, -1.0487466, -1.7710226, -1.0482562, -0.1967116, 0.1960368

Time for backsubstitution: 5.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2435
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2447
type: B, layer: 1, pos: 414
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 3181
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 3133
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2458
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 2620
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2738
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 2428
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2737
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2736
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2554
type: B, layer: 1, pos: 2672
type: B, layer: 1, pos: 2505
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3433
type: B, layer: 1, pos: 3106
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3294
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 267
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 2965
type: B, layer: 1, pos: 2975
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 2735
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 2743
type: B, layer: 1, pos: 2734
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 3278
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 3574
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2744
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3138

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2435

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0148605, upper bound: 0.0149187
time: 3.15 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0148601, upper bound: 0.0149768
time: 70.49 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -2.9133403, -2.4071496, -2.9126601, -2.4071417, -0.1411410, 0.1390793
1: -4.6378860, -3.8423033, -4.6355424, -3.8422148, -0.2641295, 0.2559513
2: -1.6610351, -1.4953283, -1.6610241, -1.4953274, -0.0575633, 0.0575794
3: -0.3807510, -0.0913813, -0.3806367, -0.0913794, -0.1549913, 0.1549913
4: -1.1093820, -0.8382995, -1.1093998, -0.8394911, -0.0401593, 0.0436909
5: -0.4605753, -0.2304851, -0.4605781, -0.2307034, -0.1614243, 0.1617236
6: -1.5973018, -1.0871391, -1.5973039, -1.0873759, -0.2301489, 0.2303926
7: 0.0766262, 0.6613971, 0.0766025, 0.6610363, -0.1695691, 0.1708652
8: -1.3490329, -0.6558459, -1.3467934, -0.6557961, -0.1832694, 0.1767869
9: -1.7731106, -1.0468752, -1.7710221, -1.0467780, -0.2035410, 0.1962351

Time for backsubstitution: 5.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2435
type: B, layer: 1, pos: 2449
type: B, layer: 1, pos: 2393
type: B, layer: 1, pos: 156
type: B, layer: 1, pos: 2419
type: B, layer: 1, pos: 3110
type: B, layer: 1, pos: 3069
type: B, layer: 1, pos: 3039
type: B, layer: 1, pos: 2447
type: B, layer: 1, pos: 414
type: B, layer: 1, pos: 2448
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 2454
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 3181
type: B, layer: 1, pos: 3502
type: B, layer: 1, pos: 773
type: B, layer: 1, pos: 3133
type: B, layer: 1, pos: 804
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 2458
type: B, layer: 1, pos: 2628
type: B, layer: 1, pos: 772
type: B, layer: 1, pos: 2620
type: B, layer: 1, pos: 2512
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2738
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 2972
type: B, layer: 1, pos: 771
type: B, layer: 1, pos: 2428
type: B, layer: 1, pos: 2981
type: B, layer: 1, pos: 2737
type: B, layer: 1, pos: 2034
type: B, layer: 1, pos: 2736
type: B, layer: 1, pos: 2540
type: B, layer: 1, pos: 2554
type: B, layer: 1, pos: 2672
type: B, layer: 1, pos: 2505
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 752
type: B, layer: 1, pos: 3433
type: B, layer: 1, pos: 3106
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 2637
type: B, layer: 1, pos: 3294
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 2028
type: B, layer: 1, pos: 267
type: B, layer: 1, pos: 747
type: B, layer: 1, pos: 2596
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2555
type: B, layer: 1, pos: 3022
type: B, layer: 1, pos: 2965
type: B, layer: 1, pos: 2975
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 2735
type: B, layer: 1, pos: 600
type: B, layer: 1, pos: 761
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 736
type: B, layer: 1, pos: 2743
type: B, layer: 1, pos: 2734
type: B, layer: 1, pos: 2613
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 3278
type: B, layer: 1, pos: 2147
type: B, layer: 1, pos: 2560
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 3574
type: B, layer: 1, pos: 778
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 809
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 897
type: B, layer: 1, pos: 899
type: B, layer: 1, pos: 2744
type: B, layer: 1, pos: 3136
type: B, layer: 1, pos: 3137
type: B, layer: 1, pos: 3138

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2435

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0149769, upper bound: 0.0149187
time: 25.05 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0149768, upper bound: 0.0149767
time: 38.31 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 68.88 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 68.88
Output dim: 4, lower bound: -0.0148605, upper bound: 0.0149187
NS_A1_B2, status: Status.VERIFIED, split count: 2, time: 68.88
Output dim: 4, lower bound: -0.0148601, upper bound: 0.0149768
NS_A2_B1, status: Status.VERIFIED, split count: 2, time: 68.88
Output dim: 4, lower bound: -0.0149769, upper bound: 0.0149187
NS_A2_B2, status: Status.VERIFIED, split count: 2, time: 68.88
Output dim: 4, lower bound: -0.0149768, upper bound: 0.0149767

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 26.09 + 288.60 = 314.69 seconds
