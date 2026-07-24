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
execution time: IAR + RelationalAnalysis = 7.86 + 447.61 = 455.47 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0902751, upper bound: 0.0902754

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2209
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3215
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2574
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3182
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3450

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2209

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0902637, upper bound: 0.0902645
time: 418.63 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0902669, upper bound: 0.0902653
time: 13.11 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 431.81 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 431.81
Output dim: 3, lower bound: -0.0902637, upper bound: 0.0902645
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 431.81
Output dim: 3, lower bound: -0.0902669, upper bound: 0.0902653

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.7537891, 1.3148727, 0.7537891, 1.3148727, -0.1536247, 0.1536247
1: -5.2188272, -3.8104343, -5.2188272, -3.8104343, -0.6289933, 0.6289933
2: 0.0021258, 0.3012948, 0.0021258, 0.3012948, -0.2053427, 0.2053427
3: -0.9232672, -0.4323110, -0.9232672, -0.4323110, -0.4909562, 0.4909562
4: -1.8312688, -1.4239975, -1.8312688, -1.4239975, -0.2392879, 0.2392880
5: -0.4231189, 0.2316254, -0.4231189, 0.2316254, -0.5339804, 0.5339804
6: -3.7900395, -2.9131515, -3.7900395, -2.9131515, -0.1199851, 0.1199851
7: -1.4844670, -0.4906130, -1.4844670, -0.4906130, -0.6526780, 0.6526780
8: -3.9166214, -2.4625170, -3.9166214, -2.4625170, -0.6334481, 0.6334479
9: -5.1894259, -3.7734914, -5.1894259, -3.7734914, -0.7017405, 0.7017403

Time for backsubstitution: 5.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3215
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2574
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3182
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3450

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2210

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0902473, upper bound: 0.0902640
time: 183.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0902588, upper bound: 0.0902489
time: 312.77 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.7537891, 1.3148727, 0.7537891, 1.3148727, -0.1536247, 0.1536247
1: -5.2188272, -3.8104343, -5.2188272, -3.8104343, -0.6289933, 0.6289933
2: 0.0021258, 0.3012948, 0.0021258, 0.3012948, -0.2053427, 0.2053427
3: -0.9232672, -0.4323110, -0.9232672, -0.4323110, -0.4909562, 0.4909562
4: -1.8312688, -1.4239975, -1.8312688, -1.4239975, -0.2392879, 0.2392879
5: -0.4231189, 0.2316254, -0.4231189, 0.2316254, -0.5339804, 0.5339804
6: -3.7900395, -2.9131515, -3.7900395, -2.9131515, -0.1199851, 0.1199851
7: -1.4844670, -0.4906130, -1.4844670, -0.4906130, -0.6526780, 0.6526780
8: -3.9166214, -2.4625170, -3.9166214, -2.4625170, -0.6334481, 0.6334479
9: -5.1894259, -3.7734914, -5.1894259, -3.7734914, -0.7017400, 0.7017403

Time for backsubstitution: 5.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2210
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3215
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2574
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3182
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3450

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 2210

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0902474, upper bound: 0.0902595
time: 20.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0902637, upper bound: 0.0902491
time: 214.85 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 241.26 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 241.26
Output dim: 3, lower bound: -0.0902473, upper bound: 0.0902640
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 241.26
Output dim: 3, lower bound: -0.0902588, upper bound: 0.0902489
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 241.26
Output dim: 3, lower bound: -0.0902474, upper bound: 0.0902595
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 241.26
Output dim: 3, lower bound: -0.0902637, upper bound: 0.0902491

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.7537891, 1.3148727, 0.7537891, 1.3148727, -0.1536207, 0.1536205
1: -5.2188272, -3.8104343, -5.2188272, -3.8104343, -0.6289768, 0.6289771
2: 0.0021258, 0.3012948, 0.0021258, 0.3012948, -0.2053422, 0.2053422
3: -0.9232672, -0.4323110, -0.9232672, -0.4323110, -0.4909562, 0.4909562
4: -1.8312688, -1.4239975, -1.8312688, -1.4239975, -0.2392874, 0.2392874
5: -0.4231189, 0.2316254, -0.4231189, 0.2316254, -0.5339801, 0.5339801
6: -3.7900395, -2.9131515, -3.7900395, -2.9131515, -0.1199850, 0.1199850
7: -1.4844670, -0.4906130, -1.4844670, -0.4906130, -0.6526811, 0.6526811
8: -3.9166214, -2.4625170, -3.9166214, -2.4625170, -0.6334362, 0.6334362
9: -5.1894259, -3.7734914, -5.1894259, -3.7734914, -0.7017260, 0.7017255

Time for backsubstitution: 6.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2199
type: DSZ, layer: 1, pos: 2184
type: DSZ, layer: 1, pos: 2228
type: DSZ, layer: 1, pos: 2183
type: DSZ, layer: 1, pos: 2211
type: DSZ, layer: 1, pos: 2230
type: DSZ, layer: 1, pos: 859
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2231
type: DSZ, layer: 1, pos: 844
type: DSZ, layer: 1, pos: 848
type: DSZ, layer: 1, pos: 2208
type: DSZ, layer: 1, pos: 879
type: DSZ, layer: 1, pos: 2223
type: DSZ, layer: 1, pos: 2207
type: DSZ, layer: 1, pos: 2193
type: DSZ, layer: 1, pos: 2178
type: DSZ, layer: 1, pos: 850
type: DSZ, layer: 1, pos: 2192
type: DSZ, layer: 1, pos: 2222
type: DSZ, layer: 1, pos: 2177
type: DSZ, layer: 1, pos: 829
type: DSZ, layer: 1, pos: 830
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 843
type: DSZ, layer: 1, pos: 355
type: DSZ, layer: 1, pos: 2559
type: DSZ, layer: 1, pos: 827
type: DSZ, layer: 1, pos: 828
type: DSZ, layer: 1, pos: 2172
type: DSZ, layer: 1, pos: 2206
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2191
type: DSZ, layer: 1, pos: 2190
type: DSZ, layer: 1, pos: 2176
type: DSZ, layer: 1, pos: 2220
type: DSZ, layer: 1, pos: 2625
type: DSZ, layer: 1, pos: 3255
type: DSZ, layer: 1, pos: 856
type: DSZ, layer: 1, pos: 855
type: DSZ, layer: 1, pos: 840
type: DSZ, layer: 1, pos: 825
type: DSZ, layer: 1, pos: 826
type: DSZ, layer: 1, pos: 3016
type: DSZ, layer: 1, pos: 2132
type: DSZ, layer: 1, pos: 2621
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 3266
type: DSZ, layer: 1, pos: 3025
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2333
type: DSZ, layer: 1, pos: 2332
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 100
type: DSZ, layer: 1, pos: 53
type: DSZ, layer: 1, pos: 748
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 726
type: DSZ, layer: 1, pos: 710
type: DSZ, layer: 1, pos: 22
type: DSZ, layer: 1, pos: 2546
type: DSZ, layer: 1, pos: 2302
type: DSZ, layer: 1, pos: 2049
type: DSZ, layer: 1, pos: 2764
type: DSZ, layer: 1, pos: 758
type: DSZ, layer: 1, pos: 714
type: DSZ, layer: 1, pos: 697
type: DSZ, layer: 1, pos: 807
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 681
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 680
type: DSZ, layer: 1, pos: 3011
type: DSZ, layer: 1, pos: 2591
type: DSZ, layer: 1, pos: 3033
type: DSZ, layer: 1, pos: 3465
type: DSZ, layer: 1, pos: 698
type: DSZ, layer: 1, pos: 3220
type: DSZ, layer: 1, pos: 759
type: DSZ, layer: 1, pos: 813
type: DSZ, layer: 1, pos: 2524
type: DSZ, layer: 1, pos: 2974
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 3215
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 367
type: DSZ, layer: 1, pos: 560
type: DSZ, layer: 1, pos: 685
type: DSZ, layer: 1, pos: 686
type: DSZ, layer: 1, pos: 708
type: DSZ, layer: 1, pos: 717
type: DSZ, layer: 1, pos: 718
type: DSZ, layer: 1, pos: 719
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 749
type: DSZ, layer: 1, pos: 752
type: DSZ, layer: 1, pos: 814
type: DSZ, layer: 1, pos: 838
type: DSZ, layer: 1, pos: 839
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 867
type: DSZ, layer: 1, pos: 2025
type: DSZ, layer: 1, pos: 2042
type: DSZ, layer: 1, pos: 2065
type: DSZ, layer: 1, pos: 2071
type: DSZ, layer: 1, pos: 2080
type: DSZ, layer: 1, pos: 2086
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2159
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2202
type: DSZ, layer: 1, pos: 2218
type: DSZ, layer: 1, pos: 2219
type: DSZ, layer: 1, pos: 2238
type: DSZ, layer: 1, pos: 2242
type: DSZ, layer: 1, pos: 2243
type: DSZ, layer: 1, pos: 2248
type: DSZ, layer: 1, pos: 2249
type: DSZ, layer: 1, pos: 2298
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2475
type: DSZ, layer: 1, pos: 2490
type: DSZ, layer: 1, pos: 2493
type: DSZ, layer: 1, pos: 2520
type: DSZ, layer: 1, pos: 2574
type: DSZ, layer: 1, pos: 2594
type: DSZ, layer: 1, pos: 2618
type: DSZ, layer: 1, pos: 2745
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3000
type: DSZ, layer: 1, pos: 3019
type: DSZ, layer: 1, pos: 3023
type: DSZ, layer: 1, pos: 3182
type: DSZ, layer: 1, pos: 3259
type: DSZ, layer: 1, pos: 3278
type: DSZ, layer: 1, pos: 3450

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2199

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0901910, upper bound: 0.0902614
time: 38.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0902463, upper bound: 0.0902075
time: 164.42 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 209.45 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 209.45
Output dim: 3, lower bound: -0.0901910, upper bound: 0.0902614
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 209.45
Output dim: 3, lower bound: -0.0902463, upper bound: 0.0902075
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 209.45
Output dim: 3, lower bound: -0.0902588, upper bound: 0.0902489
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 209.45
Output dim: 3, lower bound: -0.0902474, upper bound: 0.0902595
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 209.45
Output dim: 3, lower bound: -0.0902637, upper bound: 0.0902491

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 455.47 + 1384.52 = 1839.99 seconds
