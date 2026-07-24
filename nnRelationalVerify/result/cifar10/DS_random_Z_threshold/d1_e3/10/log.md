## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 10)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0494842662


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2.8778012, -1.8798937, -2.8778012, -1.8798937, -0.4812769, 0.4812769)
1: (-5.2575808, -3.8987381, -5.2575808, -3.8987381, -0.7476737, 0.7476737)
2: (-0.2341222, 0.2171213, -0.2341222, 0.2171213, -0.2815581, 0.2815581)
3: (-0.1711811, 0.1943922, -0.1711811, 0.1943922, -0.2320125, 0.2320125)
4: (-1.1848767, -0.6444813, -1.1848767, -0.6444813, -0.2053526, 0.2053526)
5: (-0.1271298, 0.0759195, -0.1271298, 0.0759195, -0.1132039, 0.1132039)
6: (-1.9064455, -1.3409237, -1.9064455, -1.3409237, -0.1264151, 0.1264151)
7: (-1.0665450, -0.6199826, -1.0665450, -0.6199826, -0.1850241, 0.1850241)
8: (-3.6844542, -2.6162214, -3.6844542, -2.6162214, -0.5836278, 0.5836278)
9: (-4.3624473, -3.0999429, -4.3624473, -3.0999429, -0.7698600, 0.7698600)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 8.07 + 24.00 = 32.07 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0495322, upper bound: 0.0495338

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3586
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 3407
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 3168
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 663
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 3183
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 3410
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 343
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 3377
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 422
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 833

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3586

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0495309, upper bound: 0.0495315
time: 124.11 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0495309, upper bound: 0.0495318
time: 139.49 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 263.61 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 263.61
Output dim: 4, lower bound: -0.0495309, upper bound: 0.0495315
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 263.61
Output dim: 4, lower bound: -0.0495309, upper bound: 0.0495318

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -2.8778012, -1.8798937, -2.8778012, -1.8798937, -0.4812769, 0.4812769
1: -5.2575808, -3.8987381, -5.2575808, -3.8987381, -0.7476737, 0.7476737
2: -0.2341222, 0.2171213, -0.2341222, 0.2171213, -0.2815581, 0.2815581
3: -0.1711811, 0.1943922, -0.1711811, 0.1943922, -0.2320125, 0.2320125
4: -1.1848767, -0.6444813, -1.1848767, -0.6444813, -0.2053526, 0.2053526
5: -0.1271298, 0.0759195, -0.1271298, 0.0759195, -0.1132039, 0.1132039
6: -1.9064455, -1.3409237, -1.9064455, -1.3409237, -0.1264151, 0.1264151
7: -1.0665450, -0.6199826, -1.0665450, -0.6199826, -0.1850241, 0.1850241
8: -3.6844542, -2.6162214, -3.6844542, -2.6162214, -0.5836278, 0.5836278
9: -4.3624473, -3.0999429, -4.3624473, -3.0999429, -0.7698600, 0.7698600

Time for backsubstitution: 6.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 3168
type: DSZ, layer: 1, pos: 3183
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 343
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 663
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 3410
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 3407
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3377
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 422
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2389

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 160

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0494484, upper bound: 0.0495322
time: 36.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0495305, upper bound: 0.0494505
time: 62.61 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -2.8778012, -1.8798937, -2.8778012, -1.8798937, -0.4812769, 0.4812769
1: -5.2575808, -3.8987381, -5.2575808, -3.8987381, -0.7476737, 0.7476737
2: -0.2341222, 0.2171213, -0.2341222, 0.2171213, -0.2815581, 0.2815581
3: -0.1711811, 0.1943922, -0.1711811, 0.1943922, -0.2320125, 0.2320125
4: -1.1848767, -0.6444813, -1.1848767, -0.6444813, -0.2053526, 0.2053526
5: -0.1271298, 0.0759195, -0.1271298, 0.0759195, -0.1132039, 0.1132039
6: -1.9064455, -1.3409237, -1.9064455, -1.3409237, -0.1264151, 0.1264151
7: -1.0665450, -0.6199826, -1.0665450, -0.6199826, -0.1850241, 0.1850241
8: -3.6844542, -2.6162214, -3.6844542, -2.6162214, -0.5836278, 0.5836278
9: -4.3624473, -3.0999429, -4.3624473, -3.0999429, -0.7698600, 0.7698600

Time for backsubstitution: 6.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 663
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 3183
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 3377
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 3410
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 422
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 343
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 3407
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 3168
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2451

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 2204

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0495309, upper bound: 0.0495336
time: 8.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0495309, upper bound: 0.0495326
time: 119.70 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 134.48 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 134.48
Output dim: 4, lower bound: -0.0494484, upper bound: 0.0495322
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 134.48
Output dim: 4, lower bound: -0.0495305, upper bound: 0.0494505
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 134.48
Output dim: 4, lower bound: -0.0495309, upper bound: 0.0495336
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 134.48
Output dim: 4, lower bound: -0.0495309, upper bound: 0.0495326

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.8778012, -1.8798937, -2.8778012, -1.8798937, -0.4801982, 0.4800384
1: -5.2575808, -3.8987381, -5.2575808, -3.8987381, -0.7459636, 0.7456777
2: -0.2341222, 0.2171213, -0.2341222, 0.2171213, -0.2815164, 0.2815173
3: -0.1711811, 0.1943922, -0.1711811, 0.1943922, -0.2320015, 0.2320001
4: -1.1848767, -0.6444813, -1.1848767, -0.6444813, -0.2052817, 0.2052824
5: -0.1271298, 0.0759195, -0.1271298, 0.0759195, -0.1131731, 0.1131721
6: -1.9064455, -1.3409237, -1.9064455, -1.3409237, -0.1264150, 0.1264150
7: -1.0665450, -0.6199826, -1.0665450, -0.6199826, -0.1847599, 0.1847925
8: -3.6844542, -2.6162214, -3.6844542, -2.6162214, -0.5819858, 0.5816898
9: -4.3624473, -3.0999429, -4.3624473, -3.0999429, -0.7685572, 0.7683136

Time for backsubstitution: 6.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 3183
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 3410
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 343
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 3407
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 3168
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 422
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 663
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 3377
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 818

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0494213, upper bound: 0.0494224
time: 113.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0494485, upper bound: 0.0495047
time: 153.47 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.8778012, -1.8798937, -2.8778012, -1.8798937, -0.4800385, 0.4801982
1: -5.2575808, -3.8987381, -5.2575808, -3.8987381, -0.7456777, 0.7459636
2: -0.2341222, 0.2171213, -0.2341222, 0.2171213, -0.2815173, 0.2815164
3: -0.1711811, 0.1943922, -0.1711811, 0.1943922, -0.2320000, 0.2320014
4: -1.1848767, -0.6444813, -1.1848767, -0.6444813, -0.2052824, 0.2052816
5: -0.1271298, 0.0759195, -0.1271298, 0.0759195, -0.1131721, 0.1131731
6: -1.9064455, -1.3409237, -1.9064455, -1.3409237, -0.1264150, 0.1264150
7: -1.0665450, -0.6199826, -1.0665450, -0.6199826, -0.1847925, 0.1847599
8: -3.6844542, -2.6162214, -3.6844542, -2.6162214, -0.5816898, 0.5819858
9: -4.3624473, -3.0999429, -4.3624473, -3.0999429, -0.7683136, 0.7685571

Time for backsubstitution: 6.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 3410
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 422
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 3183
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 3377
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 3168
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 343
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 663
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 3407
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 847

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 453

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0495199, upper bound: 0.0494387
time: 57.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0495306, upper bound: 0.0494383
time: 125.87 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.8778012, -1.8798937, -2.8778012, -1.8798937, -0.4812769, 0.4812769
1: -5.2575808, -3.8987381, -5.2575808, -3.8987381, -0.7476737, 0.7476737
2: -0.2341222, 0.2171213, -0.2341222, 0.2171213, -0.2815581, 0.2815581
3: -0.1711811, 0.1943922, -0.1711811, 0.1943922, -0.2320125, 0.2320125
4: -1.1848767, -0.6444813, -1.1848767, -0.6444813, -0.2053526, 0.2053526
5: -0.1271298, 0.0759195, -0.1271298, 0.0759195, -0.1132039, 0.1132039
6: -1.9064455, -1.3409237, -1.9064455, -1.3409237, -0.1264151, 0.1264151
7: -1.0665450, -0.6199826, -1.0665450, -0.6199826, -0.1850241, 0.1850241
8: -3.6844542, -2.6162214, -3.6844542, -2.6162214, -0.5836278, 0.5836278
9: -4.3624473, -3.0999429, -4.3624473, -3.0999429, -0.7698600, 0.7698600

Time for backsubstitution: 6.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 3410
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 343
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 3377
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 663
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 3168
type: DSZ, layer: 1, pos: 3407
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 3183
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 422
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 817

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0494062, upper bound: 0.0495328
time: 22.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0495302, upper bound: 0.0494093
time: 6.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.8778012, -1.8798937, -2.8778012, -1.8798937, -0.4812769, 0.4812769
1: -5.2575808, -3.8987381, -5.2575808, -3.8987381, -0.7476737, 0.7476737
2: -0.2341222, 0.2171213, -0.2341222, 0.2171213, -0.2815581, 0.2815581
3: -0.1711811, 0.1943922, -0.1711811, 0.1943922, -0.2320125, 0.2320125
4: -1.1848767, -0.6444813, -1.1848767, -0.6444813, -0.2053526, 0.2053526
5: -0.1271298, 0.0759195, -0.1271298, 0.0759195, -0.1132039, 0.1132039
6: -1.9064455, -1.3409237, -1.9064455, -1.3409237, -0.1264151, 0.1264151
7: -1.0665450, -0.6199826, -1.0665450, -0.6199826, -0.1850241, 0.1850241
8: -3.6844542, -2.6162214, -3.6844542, -2.6162214, -0.5836278, 0.5836278
9: -4.3624473, -3.0999429, -4.3624473, -3.0999429, -0.7698600, 0.7698600

Time for backsubstitution: 6.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3407
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 3183
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 422
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 343
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 3168
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 663
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 3377
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 3410
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 127

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 443

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0495269, upper bound: 0.0495315
time: 349.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0495269, upper bound: 0.0495303
time: 4.96 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 360.53 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 360.53
Output dim: 4, lower bound: -0.0494213, upper bound: 0.0494224
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 360.53
Output dim: 4, lower bound: -0.0494485, upper bound: 0.0495047
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 360.53
Output dim: 4, lower bound: -0.0495199, upper bound: 0.0494387
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 360.53
Output dim: 4, lower bound: -0.0495306, upper bound: 0.0494383
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 360.53
Output dim: 4, lower bound: -0.0494062, upper bound: 0.0495328
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 360.53
Output dim: 4, lower bound: -0.0495302, upper bound: 0.0494093
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 360.53
Output dim: 4, lower bound: -0.0495269, upper bound: 0.0495315
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 360.53
Output dim: 4, lower bound: -0.0495269, upper bound: 0.0495303

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.8778012, -1.8798937, -2.8778012, -1.8798937, -0.4789101, 0.4787616
1: -5.2575808, -3.8987381, -5.2575808, -3.8987381, -0.7441288, 0.7439035
2: -0.2341222, 0.2171213, -0.2341222, 0.2171213, -0.2814453, 0.2814463
3: -0.1711811, 0.1943922, -0.1711811, 0.1943922, -0.2320012, 0.2319999
4: -1.1848767, -0.6444813, -1.1848767, -0.6444813, -0.2052459, 0.2052470
5: -0.1271298, 0.0759195, -0.1271298, 0.0759195, -0.1131203, 0.1131192
6: -1.9064455, -1.3409237, -1.9064455, -1.3409237, -0.1264130, 0.1264130
7: -1.0665450, -0.6199826, -1.0665450, -0.6199826, -0.1846877, 0.1847143
8: -3.6844542, -2.6162214, -3.6844542, -2.6162214, -0.5805694, 0.5803190
9: -4.3624473, -3.0999429, -4.3624473, -3.0999429, -0.7679420, 0.7677373

Time for backsubstitution: 6.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 3183
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 3410
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 343
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 3590
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 3168
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 3407
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 663
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 422
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 3377
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2604

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 3346

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0494486, upper bound: 0.0495002
time: 151.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0494434, upper bound: 0.0495038
time: 247.97 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 405.93 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 405.93
Output dim: 4, lower bound: -0.0494486, upper bound: 0.0495002
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 405.93
Output dim: 4, lower bound: -0.0494434, upper bound: 0.0495038
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 405.93
Output dim: 4, lower bound: -0.0495199, upper bound: 0.0494387
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 405.93
Output dim: 4, lower bound: -0.0495306, upper bound: 0.0494383
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 405.93
Output dim: 4, lower bound: -0.0494062, upper bound: 0.0495328
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 405.93
Output dim: 4, lower bound: -0.0495302, upper bound: 0.0494093
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 405.93
Output dim: 4, lower bound: -0.0495269, upper bound: 0.0495315
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 405.93
Output dim: 4, lower bound: -0.0495269, upper bound: 0.0495303

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 32.07 + 1767.94 = 1800.01 seconds
