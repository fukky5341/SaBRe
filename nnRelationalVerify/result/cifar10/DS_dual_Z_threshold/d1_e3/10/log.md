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
execution time: IAR + RelationalAnalysis = 7.80 + 23.40 = 31.20 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0495322, upper bound: 0.0495338

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2121
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 422
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 343
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 663
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3168
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 3183
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3377
type: DSZ, layer: 1, pos: 3407
type: DSZ, layer: 1, pos: 3410
type: DSZ, layer: 1, pos: 3586
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3590

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2121

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0495107, upper bound: 0.0495271
time: 23.30 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0495256, upper bound: 0.0495111
time: 7.08 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 30.46 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 30.46
Output dim: 4, lower bound: -0.0495107, upper bound: 0.0495271
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 30.46
Output dim: 4, lower bound: -0.0495256, upper bound: 0.0495111

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -2.8778012, -1.8798937, -2.8778012, -1.8798937, -0.4812532, 0.4812530
1: -5.2575808, -3.8987381, -5.2575808, -3.8987381, -0.7476251, 0.7476251
2: -0.2341222, 0.2171213, -0.2341222, 0.2171213, -0.2815570, 0.2815570
3: -0.1711811, 0.1943922, -0.1711811, 0.1943922, -0.2320123, 0.2320123
4: -1.1848767, -0.6444813, -1.1848767, -0.6444813, -0.2053411, 0.2053411
5: -0.1271298, 0.0759195, -0.1271298, 0.0759195, -0.1132042, 0.1132042
6: -1.9064455, -1.3409237, -1.9064455, -1.3409237, -0.1264023, 0.1264023
7: -1.0665450, -0.6199826, -1.0665450, -0.6199826, -0.1850233, 0.1850234
8: -3.6844542, -2.6162214, -3.6844542, -2.6162214, -0.5836162, 0.5836163
9: -4.3624473, -3.0999429, -4.3624473, -3.0999429, -0.7698510, 0.7698507

Time for backsubstitution: 5.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 422
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 343
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 663
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3168
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 3183
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3377
type: DSZ, layer: 1, pos: 3407
type: DSZ, layer: 1, pos: 3410
type: DSZ, layer: 1, pos: 3586
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3590

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2585

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0494915, upper bound: 0.0495077
time: 10.80 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0494914, upper bound: 0.0495069
time: 10.73 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -2.8778012, -1.8798937, -2.8778012, -1.8798937, -0.4812530, 0.4812533
1: -5.2575808, -3.8987381, -5.2575808, -3.8987381, -0.7476251, 0.7476251
2: -0.2341222, 0.2171213, -0.2341222, 0.2171213, -0.2815570, 0.2815570
3: -0.1711811, 0.1943922, -0.1711811, 0.1943922, -0.2320123, 0.2320123
4: -1.1848767, -0.6444813, -1.1848767, -0.6444813, -0.2053411, 0.2053411
5: -0.1271298, 0.0759195, -0.1271298, 0.0759195, -0.1132042, 0.1132042
6: -1.9064455, -1.3409237, -1.9064455, -1.3409237, -0.1264023, 0.1264023
7: -1.0665450, -0.6199826, -1.0665450, -0.6199826, -0.1850233, 0.1850234
8: -3.6844542, -2.6162214, -3.6844542, -2.6162214, -0.5836163, 0.5836161
9: -4.3624473, -3.0999429, -4.3624473, -3.0999429, -0.7698507, 0.7698510

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 2585
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 422
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 343
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 663
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3168
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 3183
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3377
type: DSZ, layer: 1, pos: 3407
type: DSZ, layer: 1, pos: 3410
type: DSZ, layer: 1, pos: 3586
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3590

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 2585

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0495058, upper bound: 0.0494924
time: 104.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0495061, upper bound: 0.0494934
time: 9.11 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 119.32 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 119.32
Output dim: 4, lower bound: -0.0494915, upper bound: 0.0495077
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 119.32
Output dim: 4, lower bound: -0.0494914, upper bound: 0.0495069
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 119.32
Output dim: 4, lower bound: -0.0495058, upper bound: 0.0494924
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 119.32
Output dim: 4, lower bound: -0.0495061, upper bound: 0.0494934

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.8778012, -1.8798937, -2.8778012, -1.8798937, -0.4812465, 0.4812466
1: -5.2575808, -3.8987381, -5.2575808, -3.8987381, -0.7476113, 0.7476106
2: -0.2341222, 0.2171213, -0.2341222, 0.2171213, -0.2815567, 0.2815567
3: -0.1711811, 0.1943922, -0.1711811, 0.1943922, -0.2320040, 0.2320039
4: -1.1848767, -0.6444813, -1.1848767, -0.6444813, -0.2053351, 0.2053361
5: -0.1271298, 0.0759195, -0.1271298, 0.0759195, -0.1131897, 0.1131891
6: -1.9064455, -1.3409237, -1.9064455, -1.3409237, -0.1263946, 0.1263946
7: -1.0665450, -0.6199826, -1.0665450, -0.6199826, -0.1850139, 0.1850142
8: -3.6844542, -2.6162214, -3.6844542, -2.6162214, -0.5835822, 0.5835831
9: -4.3624473, -3.0999429, -4.3624473, -3.0999429, -0.7698466, 0.7698455

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 422
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 343
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 663
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3168
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 3183
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3377
type: DSZ, layer: 1, pos: 3407
type: DSZ, layer: 1, pos: 3410
type: DSZ, layer: 1, pos: 3586
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3590

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 111

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0494414, upper bound: 0.0495070
time: 40.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0494905, upper bound: 0.0494605
time: 7.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.8778012, -1.8798937, -2.8778012, -1.8798937, -0.4812468, 0.4812462
1: -5.2575808, -3.8987381, -5.2575808, -3.8987381, -0.7476106, 0.7476112
2: -0.2341222, 0.2171213, -0.2341222, 0.2171213, -0.2815567, 0.2815567
3: -0.1711811, 0.1943922, -0.1711811, 0.1943922, -0.2320039, 0.2320040
4: -1.1848767, -0.6444813, -1.1848767, -0.6444813, -0.2053361, 0.2053351
5: -0.1271298, 0.0759195, -0.1271298, 0.0759195, -0.1131891, 0.1131896
6: -1.9064455, -1.3409237, -1.9064455, -1.3409237, -0.1263946, 0.1263947
7: -1.0665450, -0.6199826, -1.0665450, -0.6199826, -0.1850142, 0.1850139
8: -3.6844542, -2.6162214, -3.6844542, -2.6162214, -0.5835829, 0.5835823
9: -4.3624473, -3.0999429, -4.3624473, -3.0999429, -0.7698458, 0.7698462

Time for backsubstitution: 5.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 422
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 343
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 663
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3168
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 3183
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3377
type: DSZ, layer: 1, pos: 3407
type: DSZ, layer: 1, pos: 3410
type: DSZ, layer: 1, pos: 3586
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3590

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 111

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0494437, upper bound: 0.0495056
time: 130.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0494903, upper bound: 0.0494531
time: 50.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.8778012, -1.8798937, -2.8778012, -1.8798937, -0.4812462, 0.4812468
1: -5.2575808, -3.8987381, -5.2575808, -3.8987381, -0.7476112, 0.7476106
2: -0.2341222, 0.2171213, -0.2341222, 0.2171213, -0.2815567, 0.2815567
3: -0.1711811, 0.1943922, -0.1711811, 0.1943922, -0.2320040, 0.2320039
4: -1.1848767, -0.6444813, -1.1848767, -0.6444813, -0.2053351, 0.2053361
5: -0.1271298, 0.0759195, -0.1271298, 0.0759195, -0.1131897, 0.1131891
6: -1.9064455, -1.3409237, -1.9064455, -1.3409237, -0.1263947, 0.1263946
7: -1.0665450, -0.6199826, -1.0665450, -0.6199826, -0.1850139, 0.1850142
8: -3.6844542, -2.6162214, -3.6844542, -2.6162214, -0.5835823, 0.5835830
9: -4.3624473, -3.0999429, -4.3624473, -3.0999429, -0.7698463, 0.7698458

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 422
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 343
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 663
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3168
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 3183
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3377
type: DSZ, layer: 1, pos: 3407
type: DSZ, layer: 1, pos: 3410
type: DSZ, layer: 1, pos: 3586
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3590

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 111

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0494544, upper bound: 0.0494917
time: 171.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0495036, upper bound: 0.0494436
time: 189.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.8778012, -1.8798937, -2.8778012, -1.8798937, -0.4812466, 0.4812465
1: -5.2575808, -3.8987381, -5.2575808, -3.8987381, -0.7476106, 0.7476112
2: -0.2341222, 0.2171213, -0.2341222, 0.2171213, -0.2815567, 0.2815567
3: -0.1711811, 0.1943922, -0.1711811, 0.1943922, -0.2320039, 0.2320040
4: -1.1848767, -0.6444813, -1.1848767, -0.6444813, -0.2053361, 0.2053351
5: -0.1271298, 0.0759195, -0.1271298, 0.0759195, -0.1131891, 0.1131896
6: -1.9064455, -1.3409237, -1.9064455, -1.3409237, -0.1263946, 0.1263946
7: -1.0665450, -0.6199826, -1.0665450, -0.6199826, -0.1850142, 0.1850139
8: -3.6844542, -2.6162214, -3.6844542, -2.6162214, -0.5835831, 0.5835822
9: -4.3624473, -3.0999429, -4.3624473, -3.0999429, -0.7698455, 0.7698466

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 422
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 343
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 663
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3168
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 3183
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3377
type: DSZ, layer: 1, pos: 3407
type: DSZ, layer: 1, pos: 3410
type: DSZ, layer: 1, pos: 3586
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3590

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 111

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0494568, upper bound: 0.0494929
time: 4.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0495054, upper bound: 0.0494428
time: 14.51 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 25.32 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 25.32
Output dim: 4, lower bound: -0.0494414, upper bound: 0.0495070
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 25.32
Output dim: 4, lower bound: -0.0494905, upper bound: 0.0494605
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 25.32
Output dim: 4, lower bound: -0.0494437, upper bound: 0.0495056
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 25.32
Output dim: 4, lower bound: -0.0494903, upper bound: 0.0494531
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 25.32
Output dim: 4, lower bound: -0.0494544, upper bound: 0.0494917
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 25.32
Output dim: 4, lower bound: -0.0495036, upper bound: 0.0494436
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 25.32
Output dim: 4, lower bound: -0.0494568, upper bound: 0.0494929
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 25.32
Output dim: 4, lower bound: -0.0495054, upper bound: 0.0494428

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.8778012, -1.8798937, -2.8778012, -1.8798937, -0.4812431, 0.4812424
1: -5.2575808, -3.8987381, -5.2575808, -3.8987381, -0.7475886, 0.7475877
2: -0.2341222, 0.2171213, -0.2341222, 0.2171213, -0.2815556, 0.2815556
3: -0.1711811, 0.1943922, -0.1711811, 0.1943922, -0.2320034, 0.2320035
4: -1.1848767, -0.6444813, -1.1848767, -0.6444813, -0.2053309, 0.2053320
5: -0.1271298, 0.0759195, -0.1271298, 0.0759195, -0.1131895, 0.1131889
6: -1.9064455, -1.3409237, -1.9064455, -1.3409237, -0.1263888, 0.1263889
7: -1.0665450, -0.6199826, -1.0665450, -0.6199826, -0.1850119, 0.1850107
8: -3.6844542, -2.6162214, -3.6844542, -2.6162214, -0.5835814, 0.5835822
9: -4.3624473, -3.0999429, -4.3624473, -3.0999429, -0.7698411, 0.7698405

Time for backsubstitution: 5.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 422
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 343
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 663
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3168
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 3183
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3377
type: DSZ, layer: 1, pos: 3407
type: DSZ, layer: 1, pos: 3410
type: DSZ, layer: 1, pos: 3586
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3590

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3126

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0493801, upper bound: 0.0494894
time: 5.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0494227, upper bound: 0.0494460
time: 8.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.8778012, -1.8798937, -2.8778012, -1.8798937, -0.4812422, 0.4812433
1: -5.2575808, -3.8987381, -5.2575808, -3.8987381, -0.7475884, 0.7475880
2: -0.2341222, 0.2171213, -0.2341222, 0.2171213, -0.2815556, 0.2815556
3: -0.1711811, 0.1943922, -0.1711811, 0.1943922, -0.2320035, 0.2320035
4: -1.1848767, -0.6444813, -1.1848767, -0.6444813, -0.2053309, 0.2053319
5: -0.1271298, 0.0759195, -0.1271298, 0.0759195, -0.1131895, 0.1131889
6: -1.9064455, -1.3409237, -1.9064455, -1.3409237, -0.1263889, 0.1263888
7: -1.0665450, -0.6199826, -1.0665450, -0.6199826, -0.1850104, 0.1850121
8: -3.6844542, -2.6162214, -3.6844542, -2.6162214, -0.5835814, 0.5835823
9: -4.3624473, -3.0999429, -4.3624473, -3.0999429, -0.7698415, 0.7698400

Time for backsubstitution: 5.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 422
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 343
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 663
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3168
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 3183
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3377
type: DSZ, layer: 1, pos: 3407
type: DSZ, layer: 1, pos: 3410
type: DSZ, layer: 1, pos: 3586
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3590

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3126

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0494302, upper bound: 0.0494410
time: 43.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0494713, upper bound: 0.0493998
time: 9.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -2.8778012, -1.8798937, -2.8778012, -1.8798937, -0.4812435, 0.4812420
1: -5.2575808, -3.8987381, -5.2575808, -3.8987381, -0.7475882, 0.7475883
2: -0.2341222, 0.2171213, -0.2341222, 0.2171213, -0.2815555, 0.2815556
3: -0.1711811, 0.1943922, -0.1711811, 0.1943922, -0.2320034, 0.2320035
4: -1.1848767, -0.6444813, -1.1848767, -0.6444813, -0.2053319, 0.2053309
5: -0.1271298, 0.0759195, -0.1271298, 0.0759195, -0.1131889, 0.1131895
6: -1.9064455, -1.3409237, -1.9064455, -1.3409237, -0.1263888, 0.1263889
7: -1.0665450, -0.6199826, -1.0665450, -0.6199826, -0.1850121, 0.1850104
8: -3.6844542, -2.6162214, -3.6844542, -2.6162214, -0.5835821, 0.5835814
9: -4.3624473, -3.0999429, -4.3624473, -3.0999429, -0.7698403, 0.7698412

Time for backsubstitution: 5.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 422
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 343
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 663
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3168
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 3183
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3377
type: DSZ, layer: 1, pos: 3407
type: DSZ, layer: 1, pos: 3410
type: DSZ, layer: 1, pos: 3586
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3590

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3126

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0493827, upper bound: 0.0494857
time: 191.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0494243, upper bound: 0.0494443
time: 12.79 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -2.8778012, -1.8798937, -2.8778012, -1.8798937, -0.4812425, 0.4812429
1: -5.2575808, -3.8987381, -5.2575808, -3.8987381, -0.7475877, 0.7475887
2: -0.2341222, 0.2171213, -0.2341222, 0.2171213, -0.2815556, 0.2815556
3: -0.1711811, 0.1943922, -0.1711811, 0.1943922, -0.2320034, 0.2320035
4: -1.1848767, -0.6444813, -1.1848767, -0.6444813, -0.2053320, 0.2053309
5: -0.1271298, 0.0759195, -0.1271298, 0.0759195, -0.1131889, 0.1131895
6: -1.9064455, -1.3409237, -1.9064455, -1.3409237, -0.1263889, 0.1263888
7: -1.0665450, -0.6199826, -1.0665450, -0.6199826, -0.1850107, 0.1850118
8: -3.6844542, -2.6162214, -3.6844542, -2.6162214, -0.5835821, 0.5835814
9: -4.3624473, -3.0999429, -4.3624473, -3.0999429, -0.7698408, 0.7698407

Time for backsubstitution: 6.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 422
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 343
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 663
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3168
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 3183
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3377
type: DSZ, layer: 1, pos: 3407
type: DSZ, layer: 1, pos: 3410
type: DSZ, layer: 1, pos: 3586
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3590

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3126

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0494297, upper bound: 0.0494355
time: 281.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0494701, upper bound: 0.0493944
time: 32.01 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -2.8778012, -1.8798937, -2.8778012, -1.8798937, -0.4812429, 0.4812426
1: -5.2575808, -3.8987381, -5.2575808, -3.8987381, -0.7475886, 0.7475877
2: -0.2341222, 0.2171213, -0.2341222, 0.2171213, -0.2815556, 0.2815556
3: -0.1711811, 0.1943922, -0.1711811, 0.1943922, -0.2320034, 0.2320035
4: -1.1848767, -0.6444813, -1.1848767, -0.6444813, -0.2053309, 0.2053320
5: -0.1271298, 0.0759195, -0.1271298, 0.0759195, -0.1131895, 0.1131889
6: -1.9064455, -1.3409237, -1.9064455, -1.3409237, -0.1263888, 0.1263889
7: -1.0665450, -0.6199826, -1.0665450, -0.6199826, -0.1850119, 0.1850107
8: -3.6844542, -2.6162214, -3.6844542, -2.6162214, -0.5835815, 0.5835821
9: -4.3624473, -3.0999429, -4.3624473, -3.0999429, -0.7698408, 0.7698407

Time for backsubstitution: 6.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 422
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 343
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 663
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3168
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 3183
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3377
type: DSZ, layer: 1, pos: 3407
type: DSZ, layer: 1, pos: 3410
type: DSZ, layer: 1, pos: 3586
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3590

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3126

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0493941, upper bound: 0.0494738
time: 5.80 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0494365, upper bound: 0.0494307
time: 63.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -2.8778012, -1.8798937, -2.8778012, -1.8798937, -0.4812419, 0.4812436
1: -5.2575808, -3.8987381, -5.2575808, -3.8987381, -0.7475883, 0.7475881
2: -0.2341222, 0.2171213, -0.2341222, 0.2171213, -0.2815556, 0.2815556
3: -0.1711811, 0.1943922, -0.1711811, 0.1943922, -0.2320035, 0.2320035
4: -1.1848767, -0.6444813, -1.1848767, -0.6444813, -0.2053309, 0.2053319
5: -0.1271298, 0.0759195, -0.1271298, 0.0759195, -0.1131895, 0.1131889
6: -1.9064455, -1.3409237, -1.9064455, -1.3409237, -0.1263889, 0.1263888
7: -1.0665450, -0.6199826, -1.0665450, -0.6199826, -0.1850104, 0.1850121
8: -3.6844542, -2.6162214, -3.6844542, -2.6162214, -0.5835815, 0.5835822
9: -4.3624473, -3.0999429, -4.3624473, -3.0999429, -0.7698413, 0.7698404

Time for backsubstitution: 6.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 3126
type: DSZ, layer: 1, pos: 3125
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 2664
type: DSZ, layer: 1, pos: 2633
type: DSZ, layer: 1, pos: 2451
type: DSZ, layer: 1, pos: 127
type: DSZ, layer: 1, pos: 2450
type: DSZ, layer: 1, pos: 2616
type: DSZ, layer: 1, pos: 112
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 2679
type: DSZ, layer: 1, pos: 2678
type: DSZ, layer: 1, pos: 2089
type: DSZ, layer: 1, pos: 79
type: DSZ, layer: 1, pos: 2603
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 2106
type: DSZ, layer: 1, pos: 2666
type: DSZ, layer: 1, pos: 203
type: DSZ, layer: 1, pos: 2631
type: DSZ, layer: 1, pos: 2389
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 2216
type: DSZ, layer: 1, pos: 786
type: DSZ, layer: 1, pos: 785
type: DSZ, layer: 1, pos: 801
type: DSZ, layer: 1, pos: 142
type: DSZ, layer: 1, pos: 191
type: DSZ, layer: 1, pos: 2185
type: DSZ, layer: 1, pos: 2201
type: DSZ, layer: 1, pos: 863
type: DSZ, layer: 1, pos: 2615
type: DSZ, layer: 1, pos: 784
type: DSZ, layer: 1, pos: 2681
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 2107
type: DSZ, layer: 1, pos: 2589
type: DSZ, layer: 1, pos: 2604
type: DSZ, layer: 1, pos: 755
type: DSZ, layer: 1, pos: 202
type: DSZ, layer: 1, pos: 864
type: DSZ, layer: 1, pos: 754
type: DSZ, layer: 1, pos: 787
type: DSZ, layer: 1, pos: 158
type: DSZ, layer: 1, pos: 865
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 3214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 858
type: DSZ, layer: 1, pos: 862
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 803
type: DSZ, layer: 1, pos: 160
type: DSZ, layer: 1, pos: 847
type: DSZ, layer: 1, pos: 2667
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 866
type: DSZ, layer: 1, pos: 817
type: DSZ, layer: 1, pos: 800
type: DSZ, layer: 1, pos: 799
type: DSZ, layer: 1, pos: 98
type: DSZ, layer: 1, pos: 2217
type: DSZ, layer: 1, pos: 833
type: DSZ, layer: 1, pos: 788
type: DSZ, layer: 1, pos: 157
type: DSZ, layer: 1, pos: 2232
type: DSZ, layer: 1, pos: 851
type: DSZ, layer: 1, pos: 2638
type: DSZ, layer: 1, pos: 815
type: DSZ, layer: 1, pos: 741
type: DSZ, layer: 1, pos: 880
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 834
type: DSZ, layer: 1, pos: 2203
type: DSZ, layer: 1, pos: 2357
type: DSZ, layer: 1, pos: 832
type: DSZ, layer: 1, pos: 818
type: DSZ, layer: 1, pos: 881
type: DSZ, layer: 1, pos: 2102
type: DSZ, layer: 1, pos: 2103
type: DSZ, layer: 1, pos: 852
type: DSZ, layer: 1, pos: 163
type: DSZ, layer: 1, pos: 2355
type: DSZ, layer: 1, pos: 2566
type: DSZ, layer: 1, pos: 2551
type: DSZ, layer: 1, pos: 2762
type: DSZ, layer: 1, pos: 2565
type: DSZ, layer: 1, pos: 425
type: DSZ, layer: 1, pos: 2545
type: DSZ, layer: 1, pos: 424
type: DSZ, layer: 1, pos: 2373
type: DSZ, layer: 1, pos: 2776
type: DSZ, layer: 1, pos: 2790
type: DSZ, layer: 1, pos: 2805
type: DSZ, layer: 1, pos: 3070
type: DSZ, layer: 1, pos: 2775
type: DSZ, layer: 1, pos: 2598
type: DSZ, layer: 1, pos: 2381
type: DSZ, layer: 1, pos: 2761
type: DSZ, layer: 1, pos: 3333
type: DSZ, layer: 1, pos: 2162
type: DSZ, layer: 1, pos: 408
type: DSZ, layer: 1, pos: 130
type: DSZ, layer: 1, pos: 137
type: DSZ, layer: 1, pos: 422
type: DSZ, layer: 1, pos: 618
type: DSZ, layer: 1, pos: 3335
type: DSZ, layer: 1, pos: 589
type: DSZ, layer: 1, pos: 731
type: DSZ, layer: 1, pos: 2064
type: DSZ, layer: 1, pos: 55
type: DSZ, layer: 1, pos: 729
type: DSZ, layer: 1, pos: 2543
type: DSZ, layer: 1, pos: 363
type: DSZ, layer: 1, pos: 3288
type: DSZ, layer: 1, pos: 587
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 212
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 343
type: DSZ, layer: 1, pos: 436
type: DSZ, layer: 1, pos: 443
type: DSZ, layer: 1, pos: 453
type: DSZ, layer: 1, pos: 483
type: DSZ, layer: 1, pos: 660
type: DSZ, layer: 1, pos: 663
type: DSZ, layer: 1, pos: 667
type: DSZ, layer: 1, pos: 734
type: DSZ, layer: 1, pos: 763
type: DSZ, layer: 1, pos: 764
type: DSZ, layer: 1, pos: 822
type: DSZ, layer: 1, pos: 888
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 893
type: DSZ, layer: 1, pos: 2075
type: DSZ, layer: 1, pos: 2142
type: DSZ, layer: 1, pos: 2173
type: DSZ, layer: 1, pos: 2174
type: DSZ, layer: 1, pos: 2189
type: DSZ, layer: 1, pos: 2204
type: DSZ, layer: 1, pos: 2246
type: DSZ, layer: 1, pos: 2247
type: DSZ, layer: 1, pos: 2339
type: DSZ, layer: 1, pos: 2382
type: DSZ, layer: 1, pos: 2483
type: DSZ, layer: 1, pos: 2496
type: DSZ, layer: 1, pos: 2497
type: DSZ, layer: 1, pos: 2498
type: DSZ, layer: 1, pos: 2510
type: DSZ, layer: 1, pos: 2609
type: DSZ, layer: 1, pos: 2622
type: DSZ, layer: 1, pos: 2623
type: DSZ, layer: 1, pos: 2685
type: DSZ, layer: 1, pos: 2688
type: DSZ, layer: 1, pos: 2691
type: DSZ, layer: 1, pos: 2693
type: DSZ, layer: 1, pos: 2694
type: DSZ, layer: 1, pos: 2695
type: DSZ, layer: 1, pos: 2701
type: DSZ, layer: 1, pos: 2702
type: DSZ, layer: 1, pos: 2706
type: DSZ, layer: 1, pos: 2715
type: DSZ, layer: 1, pos: 2717
type: DSZ, layer: 1, pos: 2732
type: DSZ, layer: 1, pos: 2746
type: DSZ, layer: 1, pos: 2945
type: DSZ, layer: 1, pos: 3045
type: DSZ, layer: 1, pos: 3168
type: DSZ, layer: 1, pos: 3169
type: DSZ, layer: 1, pos: 3183
type: DSZ, layer: 1, pos: 3184
type: DSZ, layer: 1, pos: 3193
type: DSZ, layer: 1, pos: 3200
type: DSZ, layer: 1, pos: 3346
type: DSZ, layer: 1, pos: 3377
type: DSZ, layer: 1, pos: 3407
type: DSZ, layer: 1, pos: 3410
type: DSZ, layer: 1, pos: 3586
type: DSZ, layer: 1, pos: 3589
type: DSZ, layer: 1, pos: 3590

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 3126

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0494440, upper bound: 0.0493842
time: 67.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0494833, upper bound: 0.0493832
time: 378.82 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 453.17 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 453.17
Output dim: 4, lower bound: -0.0493801, upper bound: 0.0494894
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 453.17
Output dim: 4, lower bound: -0.0494227, upper bound: 0.0494460
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 453.17
Output dim: 4, lower bound: -0.0494302, upper bound: 0.0494410
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 453.17
Output dim: 4, lower bound: -0.0494713, upper bound: 0.0493998
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 453.17
Output dim: 4, lower bound: -0.0493827, upper bound: 0.0494857
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 453.17
Output dim: 4, lower bound: -0.0494243, upper bound: 0.0494443
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 453.17
Output dim: 4, lower bound: -0.0494297, upper bound: 0.0494355
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 453.17
Output dim: 4, lower bound: -0.0494701, upper bound: 0.0493944
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 453.17
Output dim: 4, lower bound: -0.0493941, upper bound: 0.0494738
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 453.17
Output dim: 4, lower bound: -0.0494365, upper bound: 0.0494307
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 453.17
Output dim: 4, lower bound: -0.0494440, upper bound: 0.0493842
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 453.17
Output dim: 4, lower bound: -0.0494833, upper bound: 0.0493832
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 453.17
Output dim: 4, lower bound: -0.0494568, upper bound: 0.0494929
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 453.17
Output dim: 4, lower bound: -0.0495054, upper bound: 0.0494428

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 31.20 + 1949.51 = 1980.71 seconds
