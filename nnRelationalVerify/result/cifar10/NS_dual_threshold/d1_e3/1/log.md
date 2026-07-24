## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 1)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0901855242


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.7537891, 1.3148727, 0.7537891, 1.3148727, -0.1536279, 0.1536279)
1: (-5.2188272, -3.8104343, -5.2188272, -3.8104343, -0.6290011, 0.6290011)
2: (0.0021258, 0.3012948, 0.0021258, 0.3012948, -0.2053432, 0.2053432)
3: (-0.9232672, -0.4323110, -0.9232672, -0.4323110, -0.4909562, 0.4909562)
4: (-1.8312688, -1.4239975, -1.8312688, -1.4239975, -0.2392879, 0.2392879)
5: (-0.4231189, 0.2316254, -0.4231189, 0.2316254, -0.5339823, 0.5339824)
6: (-3.7900395, -2.9131515, -3.7900395, -2.9131515, -0.1199853, 0.1199853)
7: (-1.4844670, -0.4906130, -1.4844670, -0.4906130, -0.6526768, 0.6526768)
8: (-3.9166214, -2.4625170, -3.9166214, -2.4625170, -0.6334510, 0.6334510)
9: (-5.1894259, -3.7734914, -5.1894259, -3.7734914, -0.7017455, 0.7017455)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.70 + 440.43 = 448.13 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0902751, upper bound: 0.0902754

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2574
type: B, layer: 1, pos: 2574
type: A, layer: 1, pos: 3023
type: B, layer: 1, pos: 3023
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: A, layer: 1, pos: 2332
type: B, layer: 1, pos: 2332
type: A, layer: 1, pos: 2132
type: B, layer: 1, pos: 2132
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 2185
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 2333
type: B, layer: 1, pos: 2333
type: A, layer: 1, pos: 2183
type: B, layer: 1, pos: 2183
type: A, layer: 1, pos: 2121
type: B, layer: 1, pos: 2121
type: A, layer: 1, pos: 2199
type: B, layer: 1, pos: 2199
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2172
type: B, layer: 1, pos: 2172
type: A, layer: 1, pos: 2546
type: B, layer: 1, pos: 2546
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2173
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 367
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 2192
type: B, layer: 1, pos: 2192
type: A, layer: 1, pos: 2193
type: B, layer: 1, pos: 2193
type: A, layer: 1, pos: 3278
type: B, layer: 1, pos: 3278
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 355
type: B, layer: 1, pos: 355
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 2764
type: B, layer: 1, pos: 2764
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 3215
type: B, layer: 1, pos: 3215
type: A, layer: 1, pos: 2178
type: B, layer: 1, pos: 2178
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 2621
type: B, layer: 1, pos: 2621
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 2211
type: B, layer: 1, pos: 2211
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 2228
type: B, layer: 1, pos: 2228
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 2177
type: B, layer: 1, pos: 2177
type: A, layer: 1, pos: 100
type: B, layer: 1, pos: 100
type: A, layer: 1, pos: 3220
type: B, layer: 1, pos: 3220
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 2591
type: B, layer: 1, pos: 2591
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 2746
type: B, layer: 1, pos: 2746
type: A, layer: 1, pos: 2745
type: B, layer: 1, pos: 2745
type: A, layer: 1, pos: 2298
type: B, layer: 1, pos: 2298
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 2559
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 3214
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 2064
type: B, layer: 1, pos: 2064
type: A, layer: 1, pos: 2210
type: B, layer: 1, pos: 2210
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 3266
type: B, layer: 1, pos: 3266
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 2209
type: B, layer: 1, pos: 2209
type: A, layer: 1, pos: 2191
type: B, layer: 1, pos: 2191
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2202
type: B, layer: 1, pos: 2202
type: A, layer: 1, pos: 2302
type: B, layer: 1, pos: 2302
type: A, layer: 1, pos: 2190
type: B, layer: 1, pos: 2190
type: A, layer: 1, pos: 2207
type: B, layer: 1, pos: 2207
type: A, layer: 1, pos: 3259
type: B, layer: 1, pos: 3259
type: A, layer: 1, pos: 2042
type: B, layer: 1, pos: 2042
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 708
type: B, layer: 1, pos: 2220
type: A, layer: 1, pos: 2220
type: B, layer: 1, pos: 2805
type: A, layer: 1, pos: 2805
type: A, layer: 1, pos: 2080
type: B, layer: 1, pos: 2080
type: A, layer: 1, pos: 2206
type: B, layer: 1, pos: 2206
type: A, layer: 1, pos: 2230
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2625
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 2231
type: B, layer: 1, pos: 2231
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 3465
type: B, layer: 1, pos: 3465
type: A, layer: 1, pos: 2065
type: B, layer: 1, pos: 2065
type: A, layer: 1, pos: 2493
type: B, layer: 1, pos: 2493
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 2223
type: B, layer: 1, pos: 2223
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 3182
type: B, layer: 1, pos: 3182
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2208
type: B, layer: 1, pos: 2208
type: A, layer: 1, pos: 3255
type: B, layer: 1, pos: 3255
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 2176
type: B, layer: 1, pos: 2176
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 2218
type: B, layer: 1, pos: 2218
type: A, layer: 1, pos: 2222
type: B, layer: 1, pos: 2222
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 879
type: B, layer: 1, pos: 879
type: A, layer: 1, pos: 2475
type: B, layer: 1, pos: 2475
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 2025
type: B, layer: 1, pos: 2025
type: A, layer: 1, pos: 3450
type: B, layer: 1, pos: 3450
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 3000
type: B, layer: 1, pos: 3000
type: A, layer: 1, pos: 2490
type: B, layer: 1, pos: 2490
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 3016
type: B, layer: 1, pos: 3016
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: A, layer: 1, pos: 2086
type: B, layer: 1, pos: 2086
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 2761
type: B, layer: 1, pos: 2761
type: A, layer: 1, pos: 2049
type: B, layer: 1, pos: 2049
type: A, layer: 1, pos: 2520
type: B, layer: 1, pos: 2520
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2594
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2594

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 2574

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0902540, upper bound: 0.0902053
time: 18.89 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0902546, upper bound: 0.0902561
time: 179.50 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 198.46 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 198.46
Output dim: 3, lower bound: -0.0902540, upper bound: 0.0902053
NS_A2, status: Status.UNKNOWN, split count: 1, time: 198.46
Output dim: 3, lower bound: -0.0902546, upper bound: 0.0902561

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.7548270, 1.3143363, 0.7547061, 1.3148715, -0.1523404, 0.1517320
1: -5.2120399, -3.8143988, -5.2127938, -3.8104362, -0.6212764, 0.6176953
2: 0.0023373, 0.3010204, 0.0021740, 0.3010550, -0.2049769, 0.2050644
3: -0.9219409, -0.4348824, -0.9231544, -0.4345897, -0.4873512, 0.4882719
4: -1.8308723, -1.4248726, -1.8312051, -1.4247638, -0.2380350, 0.2383012
5: -0.4217181, 0.2290233, -0.4230611, 0.2293213, -0.5303962, 0.5313786
6: -3.7896299, -2.9152665, -3.7900286, -2.9150286, -0.1151830, 0.1165868
7: -1.4833765, -0.4930554, -1.4843100, -0.4927683, -0.6489172, 0.6499386
8: -3.9148550, -2.4632261, -3.9150510, -2.4625187, -0.6312819, 0.6302102
9: -5.1875620, -3.7743158, -5.1877799, -3.7734914, -0.6996017, 0.6985569

Time for backsubstitution: 5.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3023
type: A, layer: 1, pos: 3023
type: B, layer: 1, pos: 3033
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 2332
type: A, layer: 1, pos: 2332
type: B, layer: 1, pos: 2132
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 2185
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 2333
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 2333
type: B, layer: 1, pos: 2183
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2121
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2199
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2172
type: B, layer: 1, pos: 2172
type: A, layer: 1, pos: 2546
type: B, layer: 1, pos: 2546
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2173
type: B, layer: 1, pos: 2173
type: A, layer: 1, pos: 367
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 2192
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2193
type: B, layer: 1, pos: 2193
type: A, layer: 1, pos: 3278
type: B, layer: 1, pos: 3278
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 355
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 2764
type: A, layer: 1, pos: 2764
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 2621
type: A, layer: 1, pos: 3215
type: B, layer: 1, pos: 3215
type: A, layer: 1, pos: 2178
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 2621
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2211
type: A, layer: 1, pos: 2211
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 2228
type: A, layer: 1, pos: 2228
type: B, layer: 1, pos: 2574
type: A, layer: 1, pos: 2177
type: B, layer: 1, pos: 100
type: B, layer: 1, pos: 2177
type: A, layer: 1, pos: 100
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 3220
type: B, layer: 1, pos: 3220
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 2591
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 2746
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 2746
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 2298
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2210
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 2559
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 3266
type: A, layer: 1, pos: 3266
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2209
type: B, layer: 1, pos: 2209
type: A, layer: 1, pos: 2191
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2190
type: B, layer: 1, pos: 2190
type: A, layer: 1, pos: 3259
type: A, layer: 1, pos: 2207
type: B, layer: 1, pos: 2207
type: A, layer: 1, pos: 2042
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2220
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2805
type: A, layer: 1, pos: 2805
type: A, layer: 1, pos: 2080
type: B, layer: 1, pos: 2080
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2206
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2230
type: A, layer: 1, pos: 2625
type: B, layer: 1, pos: 2625
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 2231
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 3465
type: B, layer: 1, pos: 3465
type: B, layer: 1, pos: 2065
type: A, layer: 1, pos: 2065
type: B, layer: 1, pos: 2493
type: A, layer: 1, pos: 2493
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 2223
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 2223
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 2208
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 3182
type: B, layer: 1, pos: 3182
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3255
type: A, layer: 1, pos: 3255
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 2176
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 2208
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 2218
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2222
type: B, layer: 1, pos: 879
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 2475
type: B, layer: 1, pos: 2475
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 2025
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 3450
type: B, layer: 1, pos: 3450
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 3000
type: B, layer: 1, pos: 3000
type: A, layer: 1, pos: 2490
type: B, layer: 1, pos: 2490
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 3016
type: A, layer: 1, pos: 3016
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2761
type: A, layer: 1, pos: 2086
type: B, layer: 1, pos: 2086
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2049
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2520
type: B, layer: 1, pos: 2520
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2594
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2594
type: A, layer: 1, pos: 2761

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3023

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0902489, upper bound: 0.0901182
time: 14.07 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0902486, upper bound: 0.0901987
time: 63.08 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.7539629, 1.3148726, 0.7539462, 1.3148730, -0.1516498, 0.1535636
1: -5.2180729, -3.8104343, -5.2181702, -3.8104348, -0.6171985, 0.6287317
2: 0.0021382, 0.3012786, 0.0021370, 0.3012806, -0.2052883, 0.2050266
3: -0.9232484, -0.4326331, -0.9232507, -0.4325931, -0.4906553, 0.4906176
4: -1.8312583, -1.4241939, -1.8312598, -1.4241709, -0.2391618, 0.2386264
5: -0.4231096, 0.2312710, -0.4231110, 0.2313119, -0.5336208, 0.5334736
6: -3.7900372, -2.9139202, -3.7900379, -2.9138279, -0.1198419, 0.1149027
7: -1.4844414, -0.4912463, -1.4844437, -0.4911786, -0.6523204, 0.6522293
8: -3.9162619, -2.4625196, -3.9163101, -2.4625185, -0.6300497, 0.6332974
9: -5.1891046, -3.7734923, -5.1891460, -3.7734923, -0.6984711, 0.7014520

Time for backsubstitution: 5.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3023
type: B, layer: 1, pos: 3023
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: A, layer: 1, pos: 2332
type: B, layer: 1, pos: 2332
type: A, layer: 1, pos: 2132
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2333
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 2333
type: A, layer: 1, pos: 2183
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2121
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 2199
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2172
type: A, layer: 1, pos: 2172
type: B, layer: 1, pos: 2546
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2546
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 2173
type: A, layer: 1, pos: 2173
type: B, layer: 1, pos: 367
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 2192
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2193
type: A, layer: 1, pos: 2193
type: B, layer: 1, pos: 3278
type: A, layer: 1, pos: 3278
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 355
type: A, layer: 1, pos: 355
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 2764
type: B, layer: 1, pos: 2764
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 3215
type: A, layer: 1, pos: 3215
type: B, layer: 1, pos: 2178
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 2178
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 2621
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2211
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 100
type: A, layer: 1, pos: 2228
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2177
type: A, layer: 1, pos: 2177
type: B, layer: 1, pos: 3220
type: A, layer: 1, pos: 3220
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 100
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 2746
type: A, layer: 1, pos: 2745
type: B, layer: 1, pos: 2745
type: A, layer: 1, pos: 2746
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 2298
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 2574
type: B, layer: 1, pos: 2064
type: A, layer: 1, pos: 2064
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 2202
type: A, layer: 1, pos: 2210
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 2302
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 3266
type: B, layer: 1, pos: 3266
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2209
type: A, layer: 1, pos: 2209
type: B, layer: 1, pos: 2191
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2190
type: A, layer: 1, pos: 2190
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 2207
type: A, layer: 1, pos: 2207
type: B, layer: 1, pos: 2042
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 3259
type: A, layer: 1, pos: 2220
type: B, layer: 1, pos: 2220
type: A, layer: 1, pos: 2805
type: B, layer: 1, pos: 2805
type: B, layer: 1, pos: 2080
type: A, layer: 1, pos: 2080
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2206
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2230
type: B, layer: 1, pos: 2625
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 2231
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 3465
type: A, layer: 1, pos: 3465
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2208
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2493
type: B, layer: 1, pos: 2065
type: A, layer: 1, pos: 2302
type: B, layer: 1, pos: 2493
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 2202
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 2223
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 3182
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 3182
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3255
type: B, layer: 1, pos: 3255
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 2176
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 2218
type: A, layer: 1, pos: 2218
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 879
type: B, layer: 1, pos: 879
type: A, layer: 1, pos: 2208
type: B, layer: 1, pos: 2475
type: A, layer: 1, pos: 2475
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 2222
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 2025
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 3450
type: A, layer: 1, pos: 3450
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 3000
type: A, layer: 1, pos: 3000
type: B, layer: 1, pos: 2490
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 2761
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 3016
type: B, layer: 1, pos: 3016
type: B, layer: 1, pos: 2071
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2049
type: B, layer: 1, pos: 2086
type: A, layer: 1, pos: 2086
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 2520
type: A, layer: 1, pos: 2520
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2594
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2594
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2559

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3023

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0901637, upper bound: 0.0902498
time: 8.33 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0902490, upper bound: 0.0902500
time: 12.38 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 26.72 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 26.72
Output dim: 3, lower bound: -0.0902489, upper bound: 0.0901182
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 26.72
Output dim: 3, lower bound: -0.0902486, upper bound: 0.0901987
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 26.72
Output dim: 3, lower bound: -0.0901637, upper bound: 0.0902498
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 26.72
Output dim: 3, lower bound: -0.0902490, upper bound: 0.0902500

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.7548270, 1.3141588, 0.7547061, 1.3146633, -0.1519538, 0.1513355
1: -5.2120399, -3.8161497, -5.2127938, -3.8125248, -0.6172507, 0.6137474
2: 0.0024651, 0.3010201, 0.0023246, 0.3010547, -0.2047507, 0.2048189
3: -0.9210700, -0.4348875, -0.9221618, -0.4345964, -0.4864736, 0.4872743
4: -1.8305790, -1.4249068, -1.8308562, -1.4248043, -0.2377622, 0.2380017
5: -0.4210821, 0.2290233, -0.4223059, 0.2293213, -0.5297389, 0.5305994
6: -3.7878966, -2.9152665, -3.7879591, -2.9150286, -0.1128398, 0.1141465
7: -1.4827459, -0.4930554, -1.4835641, -0.4927683, -0.6482422, 0.6491381
8: -3.9148555, -2.4635115, -3.9150507, -2.4628477, -0.6307485, 0.6296756
9: -5.1875300, -3.7747331, -5.1877413, -3.7739859, -0.6987200, 0.6976688

Time for backsubstitution: 5.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3033
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 2332
type: A, layer: 1, pos: 2332
type: B, layer: 1, pos: 2132
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 2185
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 2333
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 2333
type: B, layer: 1, pos: 2183
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2121
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2199
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2172
type: B, layer: 1, pos: 2172
type: A, layer: 1, pos: 2546
type: B, layer: 1, pos: 2546
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2173
type: B, layer: 1, pos: 2173
type: A, layer: 1, pos: 367
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 2192
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2193
type: B, layer: 1, pos: 2193
type: A, layer: 1, pos: 3278
type: B, layer: 1, pos: 3278
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 355
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 2764
type: A, layer: 1, pos: 2764
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 2621
type: A, layer: 1, pos: 3215
type: B, layer: 1, pos: 3215
type: A, layer: 1, pos: 2178
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 2621
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2211
type: A, layer: 1, pos: 2211
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 2228
type: A, layer: 1, pos: 2228
type: B, layer: 1, pos: 2574
type: A, layer: 1, pos: 2177
type: B, layer: 1, pos: 100
type: A, layer: 1, pos: 3023
type: B, layer: 1, pos: 2177
type: A, layer: 1, pos: 100
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 3220
type: B, layer: 1, pos: 3220
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 2591
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 2746
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 2746
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 2298
type: B, layer: 1, pos: 2298
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2210
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 2559
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 3266
type: A, layer: 1, pos: 3266
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2209
type: B, layer: 1, pos: 2209
type: A, layer: 1, pos: 2191
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2190
type: B, layer: 1, pos: 2190
type: A, layer: 1, pos: 3259
type: A, layer: 1, pos: 2207
type: B, layer: 1, pos: 2207
type: A, layer: 1, pos: 2042
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2220
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2805
type: A, layer: 1, pos: 2805
type: A, layer: 1, pos: 2080
type: B, layer: 1, pos: 2080
type: A, layer: 1, pos: 2230
type: B, layer: 1, pos: 2206
type: A, layer: 1, pos: 2206
type: B, layer: 1, pos: 2230
type: A, layer: 1, pos: 2625
type: B, layer: 1, pos: 2625
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 2231
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 3465
type: B, layer: 1, pos: 3465
type: B, layer: 1, pos: 2065
type: A, layer: 1, pos: 2065
type: B, layer: 1, pos: 2493
type: A, layer: 1, pos: 2493
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 2223
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 2223
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 2208
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 3182
type: B, layer: 1, pos: 3182
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3255
type: A, layer: 1, pos: 3255
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 2176
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 2208
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 2218
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2222
type: B, layer: 1, pos: 879
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 2475
type: B, layer: 1, pos: 2475
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 2025
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 3450
type: B, layer: 1, pos: 3450
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 3000
type: B, layer: 1, pos: 3000
type: A, layer: 1, pos: 2490
type: B, layer: 1, pos: 2490
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 3016
type: A, layer: 1, pos: 3016
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2761
type: A, layer: 1, pos: 2086
type: B, layer: 1, pos: 2086
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2049
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2520
type: B, layer: 1, pos: 2520
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2594
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2594
type: A, layer: 1, pos: 2761

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3033

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0902348, upper bound: 0.0900328
time: 135.14 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0902348, upper bound: 0.0901044
time: 17.60 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 0.7548270, 1.3139610, 0.7540804, 1.3144482, -0.1519637, 0.1529912
1: -5.2120399, -3.8179321, -5.2194567, -3.8145599, -0.6174016, 0.6304374
2: 0.0025208, 0.3010203, 0.0023441, 0.3016406, -0.2057638, 0.2048314
3: -0.9211096, -0.4348828, -0.9223242, -0.4313526, -0.4897571, 0.4874413
4: -1.8305860, -1.4248776, -1.8309451, -1.4233525, -0.2384664, 0.2379725
5: -0.4202090, 0.2290233, -0.4213960, 0.2312601, -0.5306640, 0.5295471
6: -3.7887921, -2.9152665, -3.7891440, -2.9089203, -0.1227613, 0.1143677
7: -1.4816408, -0.4930554, -1.4824502, -0.4909041, -0.6491146, 0.6479774
8: -3.9148548, -2.4637597, -3.9159634, -2.4631460, -0.6307693, 0.6318827
9: -5.1875582, -3.7753305, -5.1894102, -3.7747459, -0.6993132, 0.7013094

Time for backsubstitution: 5.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: A, layer: 1, pos: 2332
type: B, layer: 1, pos: 2332
type: A, layer: 1, pos: 2132
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2333
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 2333
type: A, layer: 1, pos: 2183
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2121
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 2199
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2172
type: A, layer: 1, pos: 2172
type: B, layer: 1, pos: 2546
type: A, layer: 1, pos: 2546
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 2173
type: A, layer: 1, pos: 2173
type: B, layer: 1, pos: 367
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 2192
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2193
type: A, layer: 1, pos: 2193
type: B, layer: 1, pos: 3278
type: A, layer: 1, pos: 3278
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 355
type: A, layer: 1, pos: 355
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 2764
type: B, layer: 1, pos: 2764
type: A, layer: 1, pos: 2621
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 3215
type: A, layer: 1, pos: 3215
type: B, layer: 1, pos: 2178
type: A, layer: 1, pos: 2178
type: A, layer: 1, pos: 848
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 2574
type: B, layer: 1, pos: 2621
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2211
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 100
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2177
type: A, layer: 1, pos: 2177
type: B, layer: 1, pos: 3220
type: A, layer: 1, pos: 3220
type: B, layer: 1, pos: 100
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 2559
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 2746
type: A, layer: 1, pos: 2745
type: B, layer: 1, pos: 2745
type: A, layer: 1, pos: 2746
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 2298
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 2064
type: A, layer: 1, pos: 2064
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 2210
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 2202
type: B, layer: 1, pos: 2302
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 3266
type: B, layer: 1, pos: 3266
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2209
type: A, layer: 1, pos: 2209
type: B, layer: 1, pos: 2191
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2190
type: A, layer: 1, pos: 2190
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 2207
type: A, layer: 1, pos: 2207
type: B, layer: 1, pos: 2042
type: A, layer: 1, pos: 2042
type: B, layer: 1, pos: 2591
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 3259
type: A, layer: 1, pos: 2220
type: B, layer: 1, pos: 2220
type: A, layer: 1, pos: 2805
type: B, layer: 1, pos: 2805
type: B, layer: 1, pos: 2080
type: A, layer: 1, pos: 2080
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2206
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2230
type: B, layer: 1, pos: 2625
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 2231
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 3465
type: A, layer: 1, pos: 3465
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2493
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2208
type: B, layer: 1, pos: 2493
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 2223
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 2223
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 3182
type: A, layer: 1, pos: 3182
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 3255
type: B, layer: 1, pos: 3255
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 2176
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 2218
type: A, layer: 1, pos: 2218
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 2208
type: B, layer: 1, pos: 879
type: B, layer: 1, pos: 2475
type: A, layer: 1, pos: 2475
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 2025
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 3450
type: A, layer: 1, pos: 3450
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 3000
type: A, layer: 1, pos: 3000
type: A, layer: 1, pos: 2761
type: B, layer: 1, pos: 2490
type: A, layer: 1, pos: 2490
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 3016
type: B, layer: 1, pos: 3016
type: B, layer: 1, pos: 2071
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2049
type: B, layer: 1, pos: 2086
type: A, layer: 1, pos: 2086
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 2520
type: A, layer: 1, pos: 2520
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2594
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2594
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2559

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3033

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.0901633, upper bound: 0.0901849
time: 147.17 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0902347, upper bound: 0.0901843
time: 388.60 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: 0.7539629, 1.3146642, 0.7539462, 1.3146980, -0.1512538, 0.1531768
1: -5.2180729, -3.8125234, -5.2181702, -3.8121810, -0.6132531, 0.6247053
2: 0.0022887, 0.3012783, 0.0022638, 0.3012802, -0.2050417, 0.2048011
3: -0.9222538, -0.4326400, -0.9223989, -0.4325987, -0.4896551, 0.4897589
4: -1.8309098, -1.4242345, -1.8309673, -1.4242049, -0.2388624, 0.2383542
5: -0.4223538, 0.2312710, -0.4224774, 0.2313119, -0.5328414, 0.5328197
6: -3.7879679, -2.9139202, -3.7883072, -2.9138279, -0.1174015, 0.1125611
7: -1.4836955, -0.4912463, -1.4838195, -0.4911786, -0.6515198, 0.6515582
8: -3.9162641, -2.4628477, -3.9163096, -2.4627974, -0.6295104, 0.6327627
9: -5.1890688, -3.7739868, -5.1891150, -3.7739058, -0.6975660, 0.7005746

Time for backsubstitution: 5.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 3033
type: A, layer: 1, pos: 2332
type: B, layer: 1, pos: 2332
type: A, layer: 1, pos: 2132
type: B, layer: 1, pos: 2132
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 146
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 3019
type: A, layer: 1, pos: 2333
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 2333
type: A, layer: 1, pos: 2183
type: B, layer: 1, pos: 2183
type: B, layer: 1, pos: 2121
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 2199
type: B, layer: 1, pos: 2199
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2172
type: A, layer: 1, pos: 2172
type: B, layer: 1, pos: 2546
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2546
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 2173
type: A, layer: 1, pos: 2173
type: B, layer: 1, pos: 367
type: A, layer: 1, pos: 367
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 2192
type: B, layer: 1, pos: 2192
type: B, layer: 1, pos: 2193
type: A, layer: 1, pos: 2193
type: B, layer: 1, pos: 3278
type: A, layer: 1, pos: 3278
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 3011
type: B, layer: 1, pos: 3011
type: B, layer: 1, pos: 355
type: A, layer: 1, pos: 355
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 850
type: A, layer: 1, pos: 2621
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 2764
type: B, layer: 1, pos: 2764
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 3215
type: A, layer: 1, pos: 3215
type: B, layer: 1, pos: 2178
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 2178
type: B, layer: 1, pos: 848
type: B, layer: 1, pos: 2621
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 843
type: B, layer: 1, pos: 843
type: B, layer: 1, pos: 560
type: A, layer: 1, pos: 560
type: A, layer: 1, pos: 2211
type: B, layer: 1, pos: 2211
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 2591
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 100
type: A, layer: 1, pos: 2228
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2177
type: A, layer: 1, pos: 2177
type: B, layer: 1, pos: 3220
type: A, layer: 1, pos: 3220
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 100
type: B, layer: 1, pos: 813
type: A, layer: 1, pos: 813
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 2618
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 827
type: B, layer: 1, pos: 827
type: B, layer: 1, pos: 2746
type: A, layer: 1, pos: 2745
type: B, layer: 1, pos: 2745
type: A, layer: 1, pos: 2746
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 2298
type: A, layer: 1, pos: 2298
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 53
type: A, layer: 1, pos: 53
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2574
type: A, layer: 1, pos: 2064
type: B, layer: 1, pos: 828
type: B, layer: 1, pos: 2202
type: A, layer: 1, pos: 2210
type: B, layer: 1, pos: 2210
type: B, layer: 1, pos: 2302
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 3266
type: B, layer: 1, pos: 3266
type: A, layer: 1, pos: 834
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 2209
type: A, layer: 1, pos: 2209
type: B, layer: 1, pos: 2191
type: A, layer: 1, pos: 2191
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2190
type: A, layer: 1, pos: 2190
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 2207
type: A, layer: 1, pos: 2207
type: B, layer: 1, pos: 2042
type: A, layer: 1, pos: 2042
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 3259
type: B, layer: 1, pos: 2220
type: A, layer: 1, pos: 2220
type: A, layer: 1, pos: 2805
type: B, layer: 1, pos: 2805
type: B, layer: 1, pos: 2080
type: A, layer: 1, pos: 2080
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2206
type: A, layer: 1, pos: 2206
type: A, layer: 1, pos: 2230
type: B, layer: 1, pos: 2625
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 2231
type: B, layer: 1, pos: 717
type: A, layer: 1, pos: 717
type: A, layer: 1, pos: 710
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 714
type: B, layer: 1, pos: 714
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 3465
type: A, layer: 1, pos: 3465
type: B, layer: 1, pos: 2591
type: B, layer: 1, pos: 2208
type: A, layer: 1, pos: 2065
type: A, layer: 1, pos: 2493
type: B, layer: 1, pos: 2065
type: B, layer: 1, pos: 2493
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 2223
type: A, layer: 1, pos: 2202
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 855
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 2223
type: A, layer: 1, pos: 680
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 759
type: B, layer: 1, pos: 3182
type: A, layer: 1, pos: 2222
type: A, layer: 1, pos: 3182
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 3255
type: B, layer: 1, pos: 3255
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 2176
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 686
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 856
type: B, layer: 1, pos: 2218
type: A, layer: 1, pos: 2218
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 879
type: B, layer: 1, pos: 879
type: A, layer: 1, pos: 2208
type: B, layer: 1, pos: 2475
type: A, layer: 1, pos: 2475
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 826
type: B, layer: 1, pos: 2222
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 2025
type: B, layer: 1, pos: 2025
type: B, layer: 1, pos: 3450
type: A, layer: 1, pos: 3450
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 3000
type: A, layer: 1, pos: 3000
type: B, layer: 1, pos: 2490
type: A, layer: 1, pos: 2490
type: A, layer: 1, pos: 2761
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 3016
type: B, layer: 1, pos: 3016
type: B, layer: 1, pos: 2071
type: A, layer: 1, pos: 2071
type: A, layer: 1, pos: 2049
type: B, layer: 1, pos: 2086
type: A, layer: 1, pos: 2086
type: B, layer: 1, pos: 698
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 2520
type: A, layer: 1, pos: 2520
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2594
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2594
type: B, layer: 1, pos: 2049
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 3023

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3033

## Relational analysis of NS_A2_A1_A1

### Relational analysis result of NS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0900775, upper bound: 0.0902352
time: 45.77 seconds

## Relational analysis of NS_A2_A1_A2

### Relational analysis result of NS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0901499, upper bound: 0.0902357
time: 100.25 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: 0.7533420, 1.3144493, 0.7539462, 1.3144970, -0.1529197, 0.1531869
1: -5.2242851, -3.8145585, -5.2181702, -3.8144379, -0.6300170, 0.6248565
2: 0.0023087, 0.3018474, 0.0023342, 0.3012805, -0.2050533, 0.2058135
3: -0.9224147, -0.4295782, -0.9224122, -0.4325944, -0.4898203, 0.4928340
4: -1.8309975, -1.4227842, -1.8309718, -1.4241757, -0.2388327, 0.2390236
5: -0.4214453, 0.2330466, -0.4214809, 0.2313119, -0.5317888, 0.5337086
6: -3.7891519, -2.9078248, -3.7891901, -2.9138279, -0.1176226, 0.1224993
7: -1.4825797, -0.4894369, -1.4826908, -0.4911786, -0.6503589, 0.6524091
8: -3.9171686, -2.4631469, -3.9163098, -2.4630537, -0.6317444, 0.6327834
9: -5.1906128, -3.7747464, -5.1891418, -3.7746291, -0.7012789, 0.7011626

Time for backsubstitution: 5.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3033
type: A, layer: 1, pos: 3033
type: B, layer: 1, pos: 2332
type: A, layer: 1, pos: 2332
type: B, layer: 1, pos: 2132
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 3019
type: B, layer: 1, pos: 2185
type: A, layer: 1, pos: 2185
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 2333
type: A, layer: 1, pos: 2333
type: B, layer: 1, pos: 2183
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2121
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2199
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2172
type: B, layer: 1, pos: 2172
type: A, layer: 1, pos: 2546
type: B, layer: 1, pos: 2546
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2173
type: B, layer: 1, pos: 2173
type: A, layer: 1, pos: 367
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 2192
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2193
type: B, layer: 1, pos: 2193
type: A, layer: 1, pos: 3278
type: B, layer: 1, pos: 3278
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 355
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 2764
type: A, layer: 1, pos: 2764
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 3215
type: B, layer: 1, pos: 3215
type: B, layer: 1, pos: 2621
type: A, layer: 1, pos: 2178
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 2621
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2574
type: B, layer: 1, pos: 2211
type: A, layer: 1, pos: 2211
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 2228
type: A, layer: 1, pos: 2228
type: B, layer: 1, pos: 3023
type: A, layer: 1, pos: 2177
type: B, layer: 1, pos: 100
type: B, layer: 1, pos: 2177
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 100
type: A, layer: 1, pos: 3220
type: B, layer: 1, pos: 3220
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 2591
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 2591
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 2746
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 2746
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 2298
type: B, layer: 1, pos: 2298
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 2064
type: A, layer: 1, pos: 2559
type: B, layer: 1, pos: 2210
type: A, layer: 1, pos: 2210
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 3266
type: A, layer: 1, pos: 3266
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2209
type: B, layer: 1, pos: 2209
type: B, layer: 1, pos: 2191
type: A, layer: 1, pos: 2191
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2190
type: A, layer: 1, pos: 2190
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2202
type: A, layer: 1, pos: 2207
type: B, layer: 1, pos: 2207
type: A, layer: 1, pos: 3259
type: A, layer: 1, pos: 2042
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: B, layer: 1, pos: 2220
type: A, layer: 1, pos: 2220
type: B, layer: 1, pos: 2805
type: A, layer: 1, pos: 2805
type: B, layer: 1, pos: 2080
type: A, layer: 1, pos: 2080
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2206
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2625
type: A, layer: 1, pos: 2625
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 2231
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 3465
type: B, layer: 1, pos: 3465
type: B, layer: 1, pos: 2065
type: A, layer: 1, pos: 2065
type: B, layer: 1, pos: 2493
type: A, layer: 1, pos: 2493
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 2223
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 2223
type: B, layer: 1, pos: 759
type: B, layer: 1, pos: 825
type: A, layer: 1, pos: 825
type: A, layer: 1, pos: 2208
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 3182
type: B, layer: 1, pos: 3182
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3255
type: A, layer: 1, pos: 3255
type: B, layer: 1, pos: 2208
type: B, layer: 1, pos: 859
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 859
type: B, layer: 1, pos: 2176
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: B, layer: 1, pos: 2222
type: A, layer: 1, pos: 2218
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2222
type: B, layer: 1, pos: 879
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 2475
type: B, layer: 1, pos: 2475
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 826
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 2025
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 3450
type: B, layer: 1, pos: 3450
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 3000
type: B, layer: 1, pos: 3000
type: A, layer: 1, pos: 2490
type: B, layer: 1, pos: 2490
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 3016
type: A, layer: 1, pos: 3016
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: A, layer: 1, pos: 2086
type: B, layer: 1, pos: 2086
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2049
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2520
type: B, layer: 1, pos: 2520
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2594
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2594
type: A, layer: 1, pos: 2761

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3033

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0902344, upper bound: 0.0901652
time: 10.44 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0902334, upper bound: 0.0902358
time: 5.94 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 22.36 seconds
NS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 22.36
Output dim: 3, lower bound: -0.0902348, upper bound: 0.0900328
NS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 22.36
Output dim: 3, lower bound: -0.0902348, upper bound: 0.0901044
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 22.36
Output dim: 3, lower bound: -0.0901633, upper bound: 0.0901849
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 22.36
Output dim: 3, lower bound: -0.0902347, upper bound: 0.0901843
NS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 22.36
Output dim: 3, lower bound: -0.0900775, upper bound: 0.0902352
NS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 22.36
Output dim: 3, lower bound: -0.0901499, upper bound: 0.0902357
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 22.36
Output dim: 3, lower bound: -0.0902344, upper bound: 0.0901652
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 22.36
Output dim: 3, lower bound: -0.0902334, upper bound: 0.0902358

## BFS NS instance: NS_A1_B1_B1

### Backsubstitution after applying NS history:
0: 0.7548269, 1.3134888, 0.7547060, 1.3138889, -0.1513017, 0.1507618
1: -5.2120399, -3.8185315, -5.2127938, -3.8152761, -0.6150131, 0.6117680
2: 0.0024805, 0.3009729, 0.0023421, 0.3009999, -0.2046153, 0.2046808
3: -0.9198961, -0.4348877, -0.9208046, -0.4345967, -0.4852994, 0.4859169
4: -1.8302658, -1.4249084, -1.8304949, -1.4248064, -0.2374726, 0.2376675
5: -0.4198324, 0.2290233, -0.4208621, 0.2293212, -0.5284672, 0.5291309
6: -3.7877266, -2.9152684, -3.7877669, -2.9150305, -0.1127013, 0.1139872
7: -1.4812292, -0.4930554, -1.4818119, -0.4927688, -0.6466670, 0.6473186
8: -3.9148550, -2.4653091, -3.9150524, -2.4649246, -0.6288745, 0.6280086
9: -5.1875305, -3.7772059, -5.1877408, -3.7768435, -0.6964335, 0.6956182

Time for backsubstitution: 5.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2332
type: A, layer: 1, pos: 2332
type: B, layer: 1, pos: 2132
type: A, layer: 1, pos: 2132
type: A, layer: 1, pos: 3019
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 3019
type: A, layer: 1, pos: 2185
type: B, layer: 1, pos: 146
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 2333
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 2333
type: B, layer: 1, pos: 2183
type: A, layer: 1, pos: 2183
type: A, layer: 1, pos: 2121
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 2199
type: A, layer: 1, pos: 2199
type: A, layer: 1, pos: 2184
type: B, layer: 1, pos: 2184
type: A, layer: 1, pos: 2172
type: B, layer: 1, pos: 2172
type: A, layer: 1, pos: 2546
type: B, layer: 1, pos: 2546
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2173
type: B, layer: 1, pos: 2173
type: A, layer: 1, pos: 367
type: B, layer: 1, pos: 367
type: B, layer: 1, pos: 758
type: A, layer: 1, pos: 758
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 829
type: A, layer: 1, pos: 829
type: B, layer: 1, pos: 833
type: A, layer: 1, pos: 833
type: B, layer: 1, pos: 2192
type: A, layer: 1, pos: 2192
type: A, layer: 1, pos: 2193
type: B, layer: 1, pos: 2193
type: A, layer: 1, pos: 3278
type: B, layer: 1, pos: 3278
type: B, layer: 1, pos: 830
type: A, layer: 1, pos: 830
type: B, layer: 1, pos: 3011
type: A, layer: 1, pos: 3011
type: A, layer: 1, pos: 355
type: B, layer: 1, pos: 355
type: B, layer: 1, pos: 814
type: A, layer: 1, pos: 814
type: B, layer: 1, pos: 850
type: A, layer: 1, pos: 850
type: B, layer: 1, pos: 2764
type: A, layer: 1, pos: 2764
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 3215
type: B, layer: 1, pos: 3215
type: B, layer: 1, pos: 2621
type: A, layer: 1, pos: 2178
type: B, layer: 1, pos: 2178
type: B, layer: 1, pos: 848
type: A, layer: 1, pos: 848
type: A, layer: 1, pos: 2621
type: B, layer: 1, pos: 752
type: A, layer: 1, pos: 752
type: B, layer: 1, pos: 843
type: A, layer: 1, pos: 843
type: A, layer: 1, pos: 560
type: B, layer: 1, pos: 560
type: B, layer: 1, pos: 2211
type: A, layer: 1, pos: 2211
type: B, layer: 1, pos: 2974
type: A, layer: 1, pos: 2974
type: B, layer: 1, pos: 838
type: A, layer: 1, pos: 838
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 2618
type: B, layer: 1, pos: 2228
type: A, layer: 1, pos: 2228
type: B, layer: 1, pos: 2574
type: A, layer: 1, pos: 3023
type: A, layer: 1, pos: 2177
type: B, layer: 1, pos: 100
type: B, layer: 1, pos: 2177
type: A, layer: 1, pos: 100
type: B, layer: 1, pos: 2618
type: A, layer: 1, pos: 3220
type: B, layer: 1, pos: 3220
type: B, layer: 1, pos: 862
type: A, layer: 1, pos: 862
type: B, layer: 1, pos: 2591
type: A, layer: 1, pos: 3033
type: A, layer: 1, pos: 813
type: B, layer: 1, pos: 813
type: B, layer: 1, pos: 726
type: A, layer: 1, pos: 726
type: B, layer: 1, pos: 852
type: A, layer: 1, pos: 852
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 2591
type: B, layer: 1, pos: 827
type: A, layer: 1, pos: 827
type: A, layer: 1, pos: 2746
type: B, layer: 1, pos: 2745
type: B, layer: 1, pos: 2746
type: A, layer: 1, pos: 2745
type: A, layer: 1, pos: 2298
type: B, layer: 1, pos: 2298
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 3214
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 53
type: B, layer: 1, pos: 53
type: B, layer: 1, pos: 2524
type: A, layer: 1, pos: 2524
type: B, layer: 1, pos: 828
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 828
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2210
type: A, layer: 1, pos: 2210
type: A, layer: 1, pos: 2559
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 3266
type: A, layer: 1, pos: 3266
type: B, layer: 1, pos: 834
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 2202
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2209
type: B, layer: 1, pos: 2209
type: A, layer: 1, pos: 2191
type: B, layer: 1, pos: 2191
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2190
type: B, layer: 1, pos: 2190
type: A, layer: 1, pos: 2207
type: A, layer: 1, pos: 3259
type: B, layer: 1, pos: 2207
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2202
type: A, layer: 1, pos: 2042
type: B, layer: 1, pos: 2042
type: B, layer: 1, pos: 3259
type: B, layer: 1, pos: 708
type: A, layer: 1, pos: 708
type: A, layer: 1, pos: 2220
type: B, layer: 1, pos: 2220
type: B, layer: 1, pos: 2805
type: A, layer: 1, pos: 2805
type: A, layer: 1, pos: 2080
type: B, layer: 1, pos: 2080
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2206
type: B, layer: 1, pos: 2206
type: B, layer: 1, pos: 2230
type: A, layer: 1, pos: 2625
type: B, layer: 1, pos: 2625
type: A, layer: 1, pos: 717
type: B, layer: 1, pos: 717
type: B, layer: 1, pos: 2231
type: B, layer: 1, pos: 710
type: A, layer: 1, pos: 710
type: A, layer: 1, pos: 2231
type: B, layer: 1, pos: 714
type: A, layer: 1, pos: 714
type: A, layer: 1, pos: 3465
type: B, layer: 1, pos: 3465
type: B, layer: 1, pos: 2065
type: A, layer: 1, pos: 2065
type: B, layer: 1, pos: 2493
type: A, layer: 1, pos: 2493
type: B, layer: 1, pos: 867
type: A, layer: 1, pos: 867
type: B, layer: 1, pos: 855
type: A, layer: 1, pos: 855
type: B, layer: 1, pos: 2223
type: A, layer: 1, pos: 759
type: A, layer: 1, pos: 2223
type: B, layer: 1, pos: 759
type: A, layer: 1, pos: 2208
type: A, layer: 1, pos: 825
type: B, layer: 1, pos: 825
type: B, layer: 1, pos: 680
type: A, layer: 1, pos: 680
type: A, layer: 1, pos: 3182
type: B, layer: 1, pos: 3182
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 3255
type: A, layer: 1, pos: 3255
type: B, layer: 1, pos: 859
type: B, layer: 1, pos: 2176
type: B, layer: 1, pos: 2208
type: A, layer: 1, pos: 2176
type: A, layer: 1, pos: 859
type: A, layer: 1, pos: 686
type: B, layer: 1, pos: 686
type: B, layer: 1, pos: 2222
type: B, layer: 1, pos: 856
type: A, layer: 1, pos: 856
type: A, layer: 1, pos: 2218
type: B, layer: 1, pos: 2218
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2222
type: B, layer: 1, pos: 879
type: A, layer: 1, pos: 879
type: A, layer: 1, pos: 2475
type: B, layer: 1, pos: 2475
type: A, layer: 1, pos: 718
type: B, layer: 1, pos: 718
type: A, layer: 1, pos: 826
type: B, layer: 1, pos: 826
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 2025
type: A, layer: 1, pos: 2025
type: A, layer: 1, pos: 3450
type: B, layer: 1, pos: 3450
type: A, layer: 1, pos: 681
type: B, layer: 1, pos: 681
type: A, layer: 1, pos: 3000
type: B, layer: 1, pos: 3000
type: A, layer: 1, pos: 2490
type: B, layer: 1, pos: 2490
type: A, layer: 1, pos: 697
type: B, layer: 1, pos: 697
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 3016
type: A, layer: 1, pos: 3016
type: A, layer: 1, pos: 2071
type: B, layer: 1, pos: 2071
type: B, layer: 1, pos: 2761
type: A, layer: 1, pos: 2086
type: B, layer: 1, pos: 2086
type: A, layer: 1, pos: 698
type: B, layer: 1, pos: 698
type: B, layer: 1, pos: 2049
type: A, layer: 1, pos: 2049
type: A, layer: 1, pos: 2520
type: B, layer: 1, pos: 2520
type: A, layer: 1, pos: 719
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 749
type: A, layer: 1, pos: 839
type: A, layer: 1, pos: 2159
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2219
type: A, layer: 1, pos: 2238
type: A, layer: 1, pos: 2242
type: A, layer: 1, pos: 2243
type: A, layer: 1, pos: 2248
type: A, layer: 1, pos: 2249
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2594
type: B, layer: 1, pos: 719
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 749
type: B, layer: 1, pos: 839
type: B, layer: 1, pos: 2159
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2219
type: B, layer: 1, pos: 2238
type: B, layer: 1, pos: 2242
type: B, layer: 1, pos: 2243
type: B, layer: 1, pos: 2248
type: B, layer: 1, pos: 2249
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2594
type: A, layer: 1, pos: 2761

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2332

## Relational analysis of NS_A1_B1_B1_B1

### Relational analysis result of NS_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0902222, upper bound: 0.0899440
time: 34.74 seconds

## Relational analysis of NS_A1_B1_B1_B2

### Relational analysis result of NS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0902221, upper bound: 0.0900208
time: 158.75 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 448.13 + 1382.33 = 1830.46 seconds
