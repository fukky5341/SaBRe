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
execution time: IAR + RelationalAnalysis = 7.81 + 23.34 = 31.15 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -0.0495322, upper bound: 0.0495338

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 425
type: A, layer: 1, pos: 424
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 343
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 3407
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 483
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 2664
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 3410
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 422
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 3335
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2717
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 2702
type: A, layer: 1, pos: 2715
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3377
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 3183
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 3333
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 363
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 3169
type: A, layer: 1, pos: 2945
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2732
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2790
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 3193
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 2701
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2805
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 2706
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 3168
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 436
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2609
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3586
type: A, layer: 1, pos: 3589
type: A, layer: 1, pos: 3590

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 425

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0495301, upper bound: 0.0494419
time: 5.84 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0495307, upper bound: 0.0495325
time: 13.01 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 18.92 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 18.92
Output dim: 4, lower bound: -0.0495301, upper bound: 0.0494419
NS_A2, status: Status.UNKNOWN, split count: 1, time: 18.92
Output dim: 4, lower bound: -0.0495307, upper bound: 0.0495325

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -2.8777728, -1.8812888, -2.8777761, -1.8811116, -0.4798595, 0.4797451
1: -5.2574244, -3.8988235, -5.2574425, -3.8988132, -0.7474670, 0.7474726
2: -0.2336427, 0.2171213, -0.2337011, 0.2171213, -0.2807503, 0.2807894
3: -0.1711001, 0.1938273, -0.1711104, 0.1938989, -0.2309652, 0.2308610
4: -1.1848311, -0.6458119, -1.1848370, -0.6456432, -0.2039914, 0.2039046
5: -0.1269744, 0.0759177, -0.1269924, 0.0759179, -0.1129567, 0.1129778
6: -1.9064369, -1.3420634, -1.9064380, -1.3419292, -0.1253056, 0.1252034
7: -1.0665344, -0.6213370, -1.0665356, -0.6211652, -0.1840468, 0.1839171
8: -3.6844161, -2.6163006, -3.6844215, -2.6162903, -0.5833978, 0.5833834
9: -4.3622313, -3.1000819, -4.3622565, -3.1000636, -0.7688010, 0.7687541

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 424
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 343
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 3407
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 483
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 2664
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 3410
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 422
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 3335
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2717
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 2702
type: B, layer: 1, pos: 2715
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3377
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 3183
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 3333
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 363
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 3169
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2790
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 3193
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 2701
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2805
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2706
type: B, layer: 1, pos: 3168
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 436
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 3586
type: B, layer: 1, pos: 3589
type: B, layer: 1, pos: 3590

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 424

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0494026, upper bound: 0.0494319
time: 51.45 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0495287, upper bound: 0.0494327
time: 9.28 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -2.8791935, -1.8798521, -2.8778012, -1.8799070, -0.4815388, 0.4802831
1: -5.2575941, -3.8987260, -5.2575798, -3.8987453, -0.7475643, 0.7476307
2: -0.2344061, 0.2177873, -0.2341109, 0.2171213, -0.2816865, 0.2816580
3: -0.1720205, 0.1944221, -0.1711806, 0.1943873, -0.2332648, 0.2314437
4: -1.1857852, -0.6444113, -1.1848750, -0.6445119, -0.2048533, 0.2044905
5: -0.1272331, 0.0761771, -0.1271239, 0.0759195, -0.1132039, 0.1132910
6: -1.9076536, -1.3408906, -1.9064456, -1.3409407, -0.1273451, 0.1255668
7: -1.0679762, -0.6199348, -1.0665436, -0.6199828, -0.1853949, 0.1841086
8: -3.6844907, -2.6162262, -3.6844418, -2.6162286, -0.5837142, 0.5834980
9: -4.3626833, -3.0997636, -4.3624105, -3.0999501, -0.7703229, 0.7693071

Time for backsubstitution: 5.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 424
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 343
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 3407
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 483
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 2664
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 3410
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 422
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 3335
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2717
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 2702
type: B, layer: 1, pos: 2715
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3377
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 3183
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 3333
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 363
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 3169
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2790
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 3193
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 2701
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2805
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 2706
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 3168
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 436
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 3586
type: B, layer: 1, pos: 3589
type: B, layer: 1, pos: 3590

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 424

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0494034, upper bound: 0.0495299
time: 142.18 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0495285, upper bound: 0.0495321
time: 4.96 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 153.10 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 153.10
Output dim: 4, lower bound: -0.0494026, upper bound: 0.0494319
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 153.10
Output dim: 4, lower bound: -0.0495287, upper bound: 0.0494327
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 153.10
Output dim: 4, lower bound: -0.0494034, upper bound: 0.0495299
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 153.10
Output dim: 4, lower bound: -0.0495285, upper bound: 0.0495321

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -2.8777721, -1.8813133, -2.8812428, -1.8810085, -0.4796605, 0.4829028
1: -5.2574158, -3.8988240, -5.2574663, -3.8984070, -0.7479517, 0.7474195
2: -0.2336366, 0.2171213, -0.2344625, 0.2185674, -0.2820067, 0.2821646
3: -0.1710989, 0.1938175, -0.1731923, 0.1939843, -0.2307745, 0.2340473
4: -1.1848305, -0.6458132, -1.1877692, -0.6454046, -0.2040936, 0.2065084
5: -0.1269705, 0.0759177, -0.1272668, 0.0765228, -0.1135286, 0.1134267
6: -1.9064370, -1.3420982, -1.9097855, -1.3418278, -0.1252571, 0.1284928
7: -1.0665343, -0.6213384, -1.0697827, -0.6210734, -0.1836015, 0.1867499
8: -3.6843951, -2.6163001, -3.6845107, -2.6162930, -0.5833181, 0.5836547
9: -4.3622203, -3.1000810, -4.3629699, -3.0993943, -0.7691814, 0.7701358

Time for backsubstitution: 5.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 343
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 3407
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 483
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 424
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 2664
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 3410
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 422
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 3335
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2717
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 2702
type: A, layer: 1, pos: 2715
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3377
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 3183
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 3333
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 363
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 3169
type: A, layer: 1, pos: 2945
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2732
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2790
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 3193
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 2701
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2805
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2706
type: A, layer: 1, pos: 3168
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 436
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2609
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3586
type: A, layer: 1, pos: 3589
type: A, layer: 1, pos: 3590

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0494053, upper bound: 0.0494328
time: 8.07 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0495295, upper bound: 0.0494328
time: 111.21 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -2.8791792, -1.8804688, -2.8777826, -1.8807073, -0.4805524, 0.4794390
1: -5.2575150, -3.8987553, -5.2574878, -3.8987846, -0.7474296, 0.7474546
2: -0.2342124, 0.2177873, -0.2338594, 0.2171213, -0.2812971, 0.2811389
3: -0.1719847, 0.1941807, -0.1711341, 0.1940742, -0.2325657, 0.2307899
4: -1.1857655, -0.6450076, -1.1848495, -0.6452786, -0.2038950, 0.2038079
5: -0.1271638, 0.0761763, -0.1270361, 0.0759184, -0.1130853, 0.1131298
6: -1.9076494, -1.3414378, -1.9064403, -1.3416483, -0.1265498, 0.1249922
7: -1.0679734, -0.6205056, -1.0665398, -0.6207233, -0.1846352, 0.1835219
8: -3.6844676, -2.6162372, -3.6844120, -2.6162426, -0.5835670, 0.5833353
9: -4.3624854, -3.0998230, -4.3621759, -3.1000249, -0.7696913, 0.7685710

Time for backsubstitution: 6.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 343
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 3407
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 483
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 424
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 2664
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 3410
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 422
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 3335
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2717
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 2702
type: A, layer: 1, pos: 2715
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3377
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 3183
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 3333
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 363
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 3169
type: A, layer: 1, pos: 2945
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2732
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2790
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 3193
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 2701
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2805
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2706
type: A, layer: 1, pos: 3168
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 436
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2609
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3586
type: A, layer: 1, pos: 3589
type: A, layer: 1, pos: 3590

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0492774, upper bound: 0.0495295
time: 32.76 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0494016, upper bound: 0.0495299
time: 37.15 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -2.8791935, -1.8798760, -2.8812687, -1.8798029, -0.4812016, 0.4831972
1: -5.2575879, -3.8987257, -5.2576084, -3.8983393, -0.7480452, 0.7475687
2: -0.2344003, 0.2177873, -0.2348711, 0.2185674, -0.2826070, 0.2827267
3: -0.1720197, 0.1944126, -0.1732621, 0.1944723, -0.2329642, 0.2344331
4: -1.1857851, -0.6444131, -1.1878072, -0.6442747, -0.2047768, 0.2069218
5: -0.1272299, 0.0761772, -0.1273974, 0.0765244, -0.1137313, 0.1137060
6: -1.9076536, -1.3409104, -1.9097931, -1.3408315, -0.1272173, 0.1287434
7: -1.0679760, -0.6199359, -1.0697899, -0.6198904, -0.1849421, 0.1869049
8: -3.6844704, -2.6162274, -3.6845331, -2.6162310, -0.5835829, 0.5837110
9: -4.3626757, -3.0997651, -4.3631306, -3.0992789, -0.7704209, 0.7703869

Time for backsubstitution: 5.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 343
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 3407
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 483
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 424
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 2664
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 3410
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 422
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 3335
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2717
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 2702
type: A, layer: 1, pos: 2715
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3377
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 3183
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 3333
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 363
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 3169
type: A, layer: 1, pos: 2945
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2732
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2790
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 3193
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 2701
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2805
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2706
type: A, layer: 1, pos: 3168
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 436
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2609
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3586
type: A, layer: 1, pos: 3589
type: A, layer: 1, pos: 3590

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0494046, upper bound: 0.0495289
time: 154.25 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0495293, upper bound: 0.0495300
time: 6.61 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 166.88 seconds
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 166.88
Output dim: 4, lower bound: -0.0494053, upper bound: 0.0494328
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 166.88
Output dim: 4, lower bound: -0.0495295, upper bound: 0.0494328
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 166.88
Output dim: 4, lower bound: -0.0492774, upper bound: 0.0495295
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 166.88
Output dim: 4, lower bound: -0.0494016, upper bound: 0.0495299
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 166.88
Output dim: 4, lower bound: -0.0494046, upper bound: 0.0495289
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 166.88
Output dim: 4, lower bound: -0.0495293, upper bound: 0.0495300

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2.8782215, -1.8813807, -2.8812425, -1.8811387, -0.4800240, 0.4828690
1: -5.2592239, -3.8993034, -5.2574663, -3.8991978, -0.7510102, 0.7474330
2: -0.2336783, 0.2171924, -0.2344344, 0.2185674, -0.2820072, 0.2822356
3: -0.1712376, 0.1937948, -0.1731870, 0.1939360, -0.2308835, 0.2340100
4: -1.1849113, -0.6439211, -1.1877645, -0.6454066, -0.2041174, 0.2083779
5: -0.1269538, 0.0759345, -0.1272345, 0.0765227, -0.1135089, 0.1134389
6: -1.9064445, -1.3420885, -1.9097850, -1.3418350, -0.1252457, 0.1285016
7: -1.0664918, -0.6213004, -1.0696795, -0.6210745, -0.1835935, 0.1868733
8: -3.6867485, -2.6162019, -3.6845093, -2.6163025, -0.5856533, 0.5836673
9: -4.3639011, -3.1004155, -4.3629646, -3.0998724, -0.7718574, 0.7702044

Time for backsubstitution: 5.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 343
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 3407
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 483
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 2664
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 3410
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 422
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 3335
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2717
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 2702
type: B, layer: 1, pos: 2715
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3377
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 3183
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 3333
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 363
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 3169
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 2790
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 3193
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 2701
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2805
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2706
type: B, layer: 1, pos: 3168
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 436
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 3586
type: B, layer: 1, pos: 3589
type: B, layer: 1, pos: 3590

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2679

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0494613, upper bound: 0.0494133
time: 133.88 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0495100, upper bound: 0.0494134
time: 113.62 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.8791792, -1.8809272, -2.8777826, -1.8811095, -0.4801469, 0.4789784
1: -5.2575140, -3.9006038, -5.2574873, -3.9004073, -0.7451242, 0.7448435
2: -0.2341204, 0.2177873, -0.2337790, 0.2171213, -0.2812135, 0.2810628
3: -0.1719750, 0.1940448, -0.1711257, 0.1939558, -0.2324260, 0.2306312
4: -1.1844608, -0.6450139, -1.1836838, -0.6452839, -0.2025779, 0.2026312
5: -0.1271186, 0.0761762, -0.1269966, 0.0759183, -0.1130418, 0.1130893
6: -1.9076490, -1.3414567, -1.9064395, -1.3416650, -0.1265346, 0.1249766
7: -1.0677795, -0.6205071, -1.0663664, -0.6207248, -0.1843513, 0.1832554
8: -3.6844633, -2.6183157, -3.6844091, -2.6180646, -0.5817180, 0.5812375
9: -4.3624744, -3.1017075, -4.3621659, -3.1017022, -0.7676152, 0.7662265

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 343
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 3407
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 483
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 2664
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 3410
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 422
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 3335
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2717
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 2702
type: B, layer: 1, pos: 2715
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3377
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 3183
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 3333
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 363
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 3169
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2790
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 3193
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 2701
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2805
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2706
type: B, layer: 1, pos: 3168
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 436
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 3586
type: B, layer: 1, pos: 3589
type: B, layer: 1, pos: 3590

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2679

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0492110, upper bound: 0.0495102
time: 21.27 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0492590, upper bound: 0.0495111
time: 7.43 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.8796284, -1.8805364, -2.8777821, -1.8808379, -0.4809179, 0.4794068
1: -5.2593226, -3.8992023, -5.2574883, -3.8995750, -0.7504880, 0.7474684
2: -0.2342545, 0.2178584, -0.2338311, 0.2171213, -0.2813008, 0.2812102
3: -0.1721234, 0.1941578, -0.1711288, 0.1940262, -0.2326752, 0.2307523
4: -1.1858721, -0.6431158, -1.1848449, -0.6452806, -0.2039371, 0.2056774
5: -0.1271469, 0.0761935, -0.1270038, 0.0759184, -0.1130693, 0.1131419
6: -1.9076571, -1.3414278, -1.9064400, -1.3416555, -0.1265387, 0.1250015
7: -1.0679374, -0.6204675, -1.0664368, -0.6207247, -0.1846519, 0.1836453
8: -3.6868198, -2.6161375, -3.6844106, -2.6162529, -0.5859023, 0.5833478
9: -4.3641653, -3.1001565, -4.3621731, -3.1005039, -0.7723671, 0.7686387

Time for backsubstitution: 5.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 343
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 3407
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 483
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 2664
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 3410
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 422
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 3335
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2717
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 2702
type: B, layer: 1, pos: 2715
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3377
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 3183
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 3333
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 363
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 3169
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 2790
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 3193
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 2701
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2805
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2706
type: B, layer: 1, pos: 3168
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 436
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 3586
type: B, layer: 1, pos: 3589
type: B, layer: 1, pos: 3590

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2679

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0493352, upper bound: 0.0495120
time: 9.66 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0493833, upper bound: 0.0495111
time: 16.05 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.8791935, -1.8803339, -2.8812685, -1.8802046, -0.4807982, 0.4827365
1: -5.2575874, -3.9005747, -5.2576075, -3.8999610, -0.7457439, 0.7449574
2: -0.2343085, 0.2177873, -0.2347912, 0.2185674, -0.2825216, 0.2826520
3: -0.1720100, 0.1942767, -0.1732538, 0.1943544, -0.2328245, 0.2342747
4: -1.1844802, -0.6444190, -1.1866604, -0.6442803, -0.2034597, 0.2057572
5: -0.1271845, 0.0761771, -0.1273578, 0.0765243, -0.1136859, 0.1136664
6: -1.9076526, -1.3409297, -1.9097927, -1.3408480, -0.1272022, 0.1287277
7: -1.0677824, -0.6199375, -1.0696248, -0.6198920, -0.1846582, 0.1866508
8: -3.6844671, -2.6183047, -3.6845310, -2.6180530, -0.5817339, 0.5816132
9: -4.3626637, -3.1016500, -4.3631201, -3.1009569, -0.7683446, 0.7680428

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 343
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 3407
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 483
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 2664
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 3410
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 422
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 3335
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2717
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 2702
type: B, layer: 1, pos: 2715
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3377
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 3183
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 3333
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 363
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 3169
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 2790
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 3193
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 2701
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2805
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2706
type: B, layer: 1, pos: 3168
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 436
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 3586
type: B, layer: 1, pos: 3589
type: B, layer: 1, pos: 3590

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 2679

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0493369, upper bound: 0.0495121
time: 6.03 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0493854, upper bound: 0.0495119
time: 6.32 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.8796427, -1.8799429, -2.8812683, -1.8799329, -0.4815664, 0.4831651
1: -5.2593970, -3.8991733, -5.2576079, -3.8991296, -0.7511036, 0.7475829
2: -0.2344423, 0.2178584, -0.2348427, 0.2185674, -0.2826082, 0.2827980
3: -0.1721585, 0.1943897, -0.1732571, 0.1944244, -0.2330738, 0.2343958
4: -1.1858916, -0.6425213, -1.1878023, -0.6442765, -0.2048190, 0.2087911
5: -0.1272129, 0.0761942, -0.1273653, 0.0765241, -0.1137127, 0.1137183
6: -1.9076612, -1.3409004, -1.9097928, -1.3408386, -0.1272061, 0.1287521
7: -1.0679402, -0.6198975, -1.0696869, -0.6198918, -0.1849588, 0.1870283
8: -3.6868231, -2.6161273, -3.6845329, -2.6162412, -0.5859182, 0.5837235
9: -4.3643546, -3.1000979, -4.3631263, -3.0997574, -0.7730966, 0.7704549

Time for backsubstitution: 6.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2679
type: B, layer: 1, pos: 2631
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 158
type: B, layer: 1, pos: 343
type: B, layer: 1, pos: 203
type: B, layer: 1, pos: 2185
type: B, layer: 1, pos: 2201
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 2666
type: B, layer: 1, pos: 3125
type: B, layer: 1, pos: 202
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 3407
type: B, layer: 1, pos: 2216
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 864
type: B, layer: 1, pos: 862
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 847
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 2450
type: B, layer: 1, pos: 589
type: B, layer: 1, pos: 2616
type: B, layer: 1, pos: 865
type: B, layer: 1, pos: 157
type: B, layer: 1, pos: 2667
type: B, layer: 1, pos: 2681
type: B, layer: 1, pos: 2615
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2451
type: B, layer: 1, pos: 483
type: B, layer: 1, pos: 834
type: B, layer: 1, pos: 866
type: B, layer: 1, pos: 851
type: B, layer: 1, pos: 142
type: B, layer: 1, pos: 832
type: B, layer: 1, pos: 833
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 2585
type: B, layer: 1, pos: 2121
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 2664
type: B, layer: 1, pos: 801
type: B, layer: 1, pos: 128
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 618
type: B, layer: 1, pos: 3410
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 425
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2389
type: B, layer: 1, pos: 786
type: B, layer: 1, pos: 880
type: B, layer: 1, pos: 2603
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 163
type: B, layer: 1, pos: 817
type: B, layer: 1, pos: 818
type: B, layer: 1, pos: 2638
type: B, layer: 1, pos: 785
type: B, layer: 1, pos: 800
type: B, layer: 1, pos: 422
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 784
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 881
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 3335
type: B, layer: 1, pos: 3126
type: B, layer: 1, pos: 815
type: B, layer: 1, pos: 453
type: B, layer: 1, pos: 3184
type: B, layer: 1, pos: 852
type: B, layer: 1, pos: 2173
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 98
type: B, layer: 1, pos: 408
type: B, layer: 1, pos: 130
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 2162
type: B, layer: 1, pos: 2103
type: B, layer: 1, pos: 3045
type: B, layer: 1, pos: 2717
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 3070
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 2373
type: B, layer: 1, pos: 2102
type: B, layer: 1, pos: 2589
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 587
type: B, layer: 1, pos: 2702
type: B, layer: 1, pos: 2715
type: B, layer: 1, pos: 799
type: B, layer: 1, pos: 3377
type: B, layer: 1, pos: 2381
type: B, layer: 1, pos: 2678
type: B, layer: 1, pos: 2623
type: B, layer: 1, pos: 803
type: B, layer: 1, pos: 788
type: B, layer: 1, pos: 2510
type: B, layer: 1, pos: 3183
type: B, layer: 1, pos: 2357
type: B, layer: 1, pos: 2776
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 2775
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 2355
type: B, layer: 1, pos: 2746
type: B, layer: 1, pos: 2622
type: B, layer: 1, pos: 2565
type: B, layer: 1, pos: 2075
type: B, layer: 1, pos: 3333
type: B, layer: 1, pos: 822
type: B, layer: 1, pos: 2217
type: B, layer: 1, pos: 2761
type: B, layer: 1, pos: 363
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 3169
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 2598
type: B, layer: 1, pos: 2732
type: B, layer: 1, pos: 3200
type: B, layer: 1, pos: 2064
type: B, layer: 1, pos: 2382
type: B, layer: 1, pos: 79
type: B, layer: 1, pos: 2790
type: B, layer: 1, pos: 2551
type: B, layer: 1, pos: 2107
type: B, layer: 1, pos: 3193
type: B, layer: 1, pos: 2089
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 729
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 2701
type: B, layer: 1, pos: 2545
type: B, layer: 1, pos: 55
type: B, layer: 1, pos: 754
type: B, layer: 1, pos: 2566
type: B, layer: 1, pos: 755
type: B, layer: 1, pos: 741
type: B, layer: 1, pos: 2805
type: B, layer: 1, pos: 2762
type: B, layer: 1, pos: 2497
type: B, layer: 1, pos: 2496
type: B, layer: 1, pos: 763
type: B, layer: 1, pos: 731
type: B, layer: 1, pos: 3214
type: B, layer: 1, pos: 2543
type: B, layer: 1, pos: 3346
type: B, layer: 1, pos: 2706
type: B, layer: 1, pos: 3168
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 212
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 216
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 220
type: B, layer: 1, pos: 436
type: B, layer: 1, pos: 443
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 663
type: B, layer: 1, pos: 667
type: B, layer: 1, pos: 734
type: B, layer: 1, pos: 764
type: B, layer: 1, pos: 888
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 893
type: B, layer: 1, pos: 2174
type: B, layer: 1, pos: 2189
type: B, layer: 1, pos: 2204
type: B, layer: 1, pos: 2246
type: B, layer: 1, pos: 2247
type: B, layer: 1, pos: 2339
type: B, layer: 1, pos: 2609
type: B, layer: 1, pos: 2685
type: B, layer: 1, pos: 2688
type: B, layer: 1, pos: 2691
type: B, layer: 1, pos: 2693
type: B, layer: 1, pos: 2694
type: B, layer: 1, pos: 2695
type: B, layer: 1, pos: 3586
type: B, layer: 1, pos: 3589
type: B, layer: 1, pos: 3590

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 2679

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0494618, upper bound: 0.0495116
time: 39.49 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0495099, upper bound: 0.0495100
time: 188.30 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 233.85 seconds
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 233.85
Output dim: 4, lower bound: -0.0494613, upper bound: 0.0494133
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 233.85
Output dim: 4, lower bound: -0.0495100, upper bound: 0.0494134
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 233.85
Output dim: 4, lower bound: -0.0492110, upper bound: 0.0495102
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 233.85
Output dim: 4, lower bound: -0.0492590, upper bound: 0.0495111
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 233.85
Output dim: 4, lower bound: -0.0493352, upper bound: 0.0495120
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 233.85
Output dim: 4, lower bound: -0.0493833, upper bound: 0.0495111
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 233.85
Output dim: 4, lower bound: -0.0493369, upper bound: 0.0495121
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 233.85
Output dim: 4, lower bound: -0.0493854, upper bound: 0.0495119
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 233.85
Output dim: 4, lower bound: -0.0494618, upper bound: 0.0495116
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 233.85
Output dim: 4, lower bound: -0.0495099, upper bound: 0.0495100

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2.8777702, -1.8813946, -2.8807311, -1.8805547, -0.4806021, 0.4825765
1: -5.2568703, -3.8993716, -5.2548428, -3.8980672, -0.7531195, 0.7460607
2: -0.2336255, 0.2171273, -0.2344510, 0.2184920, -0.2818719, 0.2821887
3: -0.1711105, 0.1937895, -0.1730595, 0.1940539, -0.2308883, 0.2338907
4: -1.1849072, -0.6444321, -1.1889946, -0.6459772, -0.2034511, 0.2089531
5: -0.1269355, 0.0758969, -0.1272818, 0.0764795, -0.1134733, 0.1134782
6: -1.9064384, -1.3420928, -1.9098331, -1.3418345, -0.1252188, 0.1285027
7: -1.0664817, -0.6215252, -1.0698879, -0.6213346, -0.1834384, 0.1870624
8: -3.6860166, -2.6162229, -3.6836789, -2.6137547, -0.5869983, 0.5825235
9: -4.3628006, -3.1004553, -4.3617377, -3.0971861, -0.7734654, 0.7689508

Time for backsubstitution: 6.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 343
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 3407
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 483
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 424
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 2664
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 3410
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 422
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 3335
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2717
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 2702
type: A, layer: 1, pos: 2715
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3377
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 3183
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 3333
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 363
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 3169
type: A, layer: 1, pos: 2945
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2732
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2790
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 3193
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 2701
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2805
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2706
type: A, layer: 1, pos: 3168
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 436
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2609
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3586
type: A, layer: 1, pos: 3589
type: A, layer: 1, pos: 3590

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -0.0493650, upper bound: 0.0494139
time: 107.11 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0495089, upper bound: 0.0494115
time: 9.11 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2.8789203, -1.8809447, -2.8774786, -1.8811297, -0.4797584, 0.4785318
1: -5.2568016, -3.9007101, -5.2566719, -3.9005277, -0.7436398, 0.7431507
2: -0.2340830, 0.2177576, -0.2337360, 0.2170864, -0.2811446, 0.2809922
3: -0.1719103, 0.1940362, -0.1710497, 0.1939455, -0.2323572, 0.2305526
4: -1.1844528, -0.6456020, -1.1836751, -0.6459573, -0.2019491, 0.2020865
5: -0.1271003, 0.0761486, -0.1269755, 0.0758863, -0.1129849, 0.1130356
6: -1.9076422, -1.3414756, -1.9064322, -1.3416870, -0.1265166, 0.1249604
7: -1.0677655, -0.6206262, -1.0663502, -0.6208642, -0.1841817, 0.1831071
8: -3.6833327, -2.6183434, -3.6830909, -2.6180978, -0.5806347, 0.5799987
9: -4.3612328, -3.1017640, -4.3607411, -3.1017675, -0.7663760, 0.7648063

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 343
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 3407
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 483
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 424
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 2664
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 3410
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 422
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 3335
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2717
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 2702
type: A, layer: 1, pos: 2715
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3377
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 3183
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 3333
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 363
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 3169
type: A, layer: 1, pos: 2945
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2732
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2790
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 3193
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 2701
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2805
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2706
type: A, layer: 1, pos: 3168
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 436
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2609
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3586
type: A, layer: 1, pos: 3589
type: A, layer: 1, pos: 3590

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0490717, upper bound: 0.0495113
time: 8.55 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0492104, upper bound: 0.0495087
time: 68.59 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2.8787286, -1.8809431, -2.8772714, -1.8805277, -0.4807278, 0.4786859
1: -5.2551594, -3.9006808, -5.2548633, -3.8992803, -0.7472336, 0.7434708
2: -0.2340672, 0.2177222, -0.2337962, 0.2170460, -0.2810781, 0.2810164
3: -0.1718479, 0.1940389, -0.1709980, 0.1940727, -0.2324314, 0.2305121
4: -1.1844556, -0.6455252, -1.1849134, -0.6458532, -0.2019127, 0.2032072
5: -0.1271004, 0.0761385, -0.1270442, 0.0758752, -0.1130062, 0.1131288
6: -1.9076424, -1.3414618, -1.9064875, -1.3416650, -0.1265082, 0.1249782
7: -1.0677694, -0.6207318, -1.0665754, -0.6209849, -0.1841962, 0.1834444
8: -3.6837323, -2.6183348, -3.6835778, -2.6155193, -0.5830637, 0.5800942
9: -4.3613739, -3.1017485, -4.3609414, -3.0990169, -0.7692204, 0.7649789

Time for backsubstitution: 5.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 2631
type: A, layer: 1, pos: 158
type: A, layer: 1, pos: 343
type: A, layer: 1, pos: 203
type: A, layer: 1, pos: 2185
type: A, layer: 1, pos: 2201
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 2666
type: A, layer: 1, pos: 3125
type: A, layer: 1, pos: 202
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 3407
type: A, layer: 1, pos: 2216
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 864
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 862
type: A, layer: 1, pos: 847
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 589
type: A, layer: 1, pos: 2450
type: A, layer: 1, pos: 2616
type: A, layer: 1, pos: 865
type: A, layer: 1, pos: 2667
type: A, layer: 1, pos: 157
type: A, layer: 1, pos: 2681
type: A, layer: 1, pos: 2615
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 483
type: A, layer: 1, pos: 2451
type: A, layer: 1, pos: 834
type: A, layer: 1, pos: 866
type: A, layer: 1, pos: 851
type: A, layer: 1, pos: 424
type: A, layer: 1, pos: 142
type: A, layer: 1, pos: 832
type: A, layer: 1, pos: 833
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 2585
type: A, layer: 1, pos: 2121
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 2664
type: A, layer: 1, pos: 801
type: A, layer: 1, pos: 128
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 618
type: A, layer: 1, pos: 3410
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2389
type: A, layer: 1, pos: 786
type: A, layer: 1, pos: 880
type: A, layer: 1, pos: 2603
type: A, layer: 1, pos: 2679
type: A, layer: 1, pos: 163
type: A, layer: 1, pos: 817
type: A, layer: 1, pos: 818
type: A, layer: 1, pos: 2638
type: A, layer: 1, pos: 785
type: A, layer: 1, pos: 422
type: A, layer: 1, pos: 800
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 784
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 881
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 3335
type: A, layer: 1, pos: 815
type: A, layer: 1, pos: 3126
type: A, layer: 1, pos: 453
type: A, layer: 1, pos: 3184
type: A, layer: 1, pos: 852
type: A, layer: 1, pos: 2173
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 98
type: A, layer: 1, pos: 408
type: A, layer: 1, pos: 130
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 2162
type: A, layer: 1, pos: 2103
type: A, layer: 1, pos: 3045
type: A, layer: 1, pos: 2717
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 3070
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 2373
type: A, layer: 1, pos: 2102
type: A, layer: 1, pos: 2589
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 587
type: A, layer: 1, pos: 2702
type: A, layer: 1, pos: 2715
type: A, layer: 1, pos: 799
type: A, layer: 1, pos: 3377
type: A, layer: 1, pos: 2381
type: A, layer: 1, pos: 2678
type: A, layer: 1, pos: 2623
type: A, layer: 1, pos: 803
type: A, layer: 1, pos: 788
type: A, layer: 1, pos: 2510
type: A, layer: 1, pos: 3183
type: A, layer: 1, pos: 2357
type: A, layer: 1, pos: 2776
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 2775
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 2355
type: A, layer: 1, pos: 2746
type: A, layer: 1, pos: 2622
type: A, layer: 1, pos: 2565
type: A, layer: 1, pos: 2075
type: A, layer: 1, pos: 3333
type: A, layer: 1, pos: 822
type: A, layer: 1, pos: 2217
type: A, layer: 1, pos: 363
type: A, layer: 1, pos: 2761
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 3169
type: A, layer: 1, pos: 2945
type: A, layer: 1, pos: 2598
type: A, layer: 1, pos: 2732
type: A, layer: 1, pos: 3200
type: A, layer: 1, pos: 2064
type: A, layer: 1, pos: 2382
type: A, layer: 1, pos: 2790
type: A, layer: 1, pos: 79
type: A, layer: 1, pos: 2551
type: A, layer: 1, pos: 2107
type: A, layer: 1, pos: 3193
type: A, layer: 1, pos: 2089
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 729
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 2701
type: A, layer: 1, pos: 2545
type: A, layer: 1, pos: 55
type: A, layer: 1, pos: 754
type: A, layer: 1, pos: 2566
type: A, layer: 1, pos: 755
type: A, layer: 1, pos: 741
type: A, layer: 1, pos: 2805
type: A, layer: 1, pos: 2762
type: A, layer: 1, pos: 2497
type: A, layer: 1, pos: 2496
type: A, layer: 1, pos: 763
type: A, layer: 1, pos: 731
type: A, layer: 1, pos: 3214
type: A, layer: 1, pos: 2543
type: A, layer: 1, pos: 3346
type: A, layer: 1, pos: 2706
type: A, layer: 1, pos: 3168
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 212
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 216
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 220
type: A, layer: 1, pos: 436
type: A, layer: 1, pos: 443
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 663
type: A, layer: 1, pos: 667
type: A, layer: 1, pos: 734
type: A, layer: 1, pos: 764
type: A, layer: 1, pos: 888
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 893
type: A, layer: 1, pos: 2174
type: A, layer: 1, pos: 2189
type: A, layer: 1, pos: 2204
type: A, layer: 1, pos: 2246
type: A, layer: 1, pos: 2247
type: A, layer: 1, pos: 2339
type: A, layer: 1, pos: 2609
type: A, layer: 1, pos: 2685
type: A, layer: 1, pos: 2688
type: A, layer: 1, pos: 2691
type: A, layer: 1, pos: 2693
type: A, layer: 1, pos: 2694
type: A, layer: 1, pos: 2695
type: A, layer: 1, pos: 3586
type: A, layer: 1, pos: 3589
type: A, layer: 1, pos: 3590

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0491135, upper bound: 0.0495108
time: 8.25 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -0.0492585, upper bound: 0.0495089
time: 412.62 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 31.15 + 1811.55 = 1842.71 seconds
