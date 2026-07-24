## Execution arguments:
Dataset: Dataset.CIFAR10
Network: ds/onnx/cifar10_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 11)
Time budget: 1800 seconds
Split limit: 100
Threshold: 0.0165089745


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.5753176, -2.7687309, -3.5753176, -2.7687309, -0.4758013, 0.4758013)
1: (-5.3830743, -4.0869861, -5.3830743, -4.0869861, -0.4215515, 0.4215515)
2: (-0.5319318, -0.3178027, -0.5319318, -0.3178027, -0.0943305, 0.0943305)
3: (-1.0287220, -0.6424292, -1.0287220, -0.6424292, -0.1155567, 0.1155567)
4: (-0.5967066, -0.0644337, -0.5967066, -0.0644337, -0.1455552, 0.1455552)
5: (-0.6392584, -0.2494622, -0.6392584, -0.2494622, -0.1566667, 0.1566667)
6: (-1.9661086, -1.2726079, -1.9661086, -1.2726079, -0.1202665, 0.1202665)
7: (0.5685282, 0.9923374, 0.5685282, 0.9923374, -0.0623855, 0.0623855)
8: (-5.5624652, -4.4831619, -5.5624652, -4.4831619, -0.3936854, 0.3936854)
9: (-4.6618037, -3.6479344, -4.6618037, -3.6479344, -0.4097505, 0.4097506)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 7.80 + 26.01 = 33.81 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0165178, upper bound: 0.0165269

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3527
type: A, layer: 1, pos: 3499
type: A, layer: 1, pos: 236
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 3526
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 3532
type: A, layer: 1, pos: 3549
type: A, layer: 1, pos: 3095
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 2875
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 2870
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2874
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3387
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 100
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2662
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2808
type: A, layer: 1, pos: 3510
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 225
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2844
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2270
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 2959
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2945
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 2938
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2333
type: A, layer: 1, pos: 330
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2944
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 2338
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 435
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 2534
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 3209
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3366
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3374
type: A, layer: 1, pos: 3419
type: A, layer: 1, pos: 3589
type: A, layer: 1, pos: 3592
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3596
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3527

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0163313, upper bound: 0.0165320
time: 6.57 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165165, upper bound: 0.0165313
time: 48.15 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 54.79 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 54.79
Output dim: 7, lower bound: -0.0163313, upper bound: 0.0165320
NS_A2, status: Status.UNKNOWN, split count: 1, time: 54.79
Output dim: 7, lower bound: -0.0165165, upper bound: 0.0165313

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3.5749140, -2.7687318, -3.5749698, -2.7687316, -0.4746778, 0.4746844
1: -5.3830709, -4.0880761, -5.3830714, -4.0879431, -0.4205728, 0.4204186
2: -0.5319080, -0.3178224, -0.5319114, -0.3178201, -0.0941545, 0.0941459
3: -1.0287166, -0.6436734, -1.0287173, -0.6435223, -0.1144512, 0.1142853
4: -0.5959620, -0.0644411, -0.5960635, -0.0644397, -0.1448011, 0.1449012
5: -0.6392423, -0.2499268, -0.6392441, -0.2498738, -0.1562384, 0.1561780
6: -1.9661084, -1.2746170, -1.9661088, -1.2743422, -0.1185286, 0.1182533
7: 0.5705536, 0.9923306, 0.5702822, 0.9923318, -0.0603450, 0.0606239
8: -5.5619941, -4.4832535, -5.5620580, -4.4832430, -0.3931993, 0.3932520
9: -4.6616240, -3.6496475, -4.6616483, -3.6494136, -0.4082204, 0.4079905

Time for backsubstitution: 6.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3499
type: B, layer: 1, pos: 236
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3526
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 3532
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 3095
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 3272
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2875
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 2870
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2874
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 100
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2662
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2975
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2808
type: B, layer: 1, pos: 3510
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 225
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2844
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 2270
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 2959
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 3525
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2858
type: B, layer: 1, pos: 2938
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2333
type: B, layer: 1, pos: 330
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2944
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 435
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 2534
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 3209
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3366
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3374
type: B, layer: 1, pos: 3419
type: B, layer: 1, pos: 3589
type: B, layer: 1, pos: 3592
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3596
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3499

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0161741, upper bound: 0.0165292
time: 42.01 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0163302, upper bound: 0.0165249
time: 4.53 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -3.5754178, -2.7684789, -3.5752382, -2.7687309, -0.4761642, 0.4754153
1: -5.3849640, -4.0870266, -5.3830748, -4.0870214, -0.4235057, 0.4205065
2: -0.5319662, -0.3178057, -0.5319315, -0.3178055, -0.0944899, 0.0941864
3: -1.0308781, -0.6424125, -1.0287213, -0.6424551, -0.1177385, 0.1144706
4: -0.5967098, -0.0631318, -0.5966984, -0.0644338, -0.1449305, 0.1468471
5: -0.6400650, -0.2494721, -0.6392579, -0.2494768, -0.1574866, 0.1562914
6: -1.9696542, -1.2726082, -1.9661092, -1.2726089, -0.1238073, 0.1184371
7: 0.5684702, 0.9959008, 0.5685472, 0.9923375, -0.0606382, 0.0659496
8: -5.5624571, -4.4823723, -5.5624580, -4.4831638, -0.3934482, 0.3944429
9: -4.6647472, -3.6479650, -4.6618042, -3.6479650, -0.4127460, 0.4086211

Time for backsubstitution: 5.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3499
type: B, layer: 1, pos: 236
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3526
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 3532
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 3095
type: B, layer: 1, pos: 3272
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2875
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 2870
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2874
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 100
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2662
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2975
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2808
type: B, layer: 1, pos: 3510
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 225
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2844
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 2270
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 2959
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 3525
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2858
type: B, layer: 1, pos: 2938
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2333
type: B, layer: 1, pos: 330
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2944
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 435
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 2534
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 3209
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3366
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3374
type: B, layer: 1, pos: 3419
type: B, layer: 1, pos: 3589
type: B, layer: 1, pos: 3592
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3596
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3499

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0163592, upper bound: 0.0165301
time: 11.09 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165159, upper bound: 0.0165232
time: 18.87 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 36.02 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 36.02
Output dim: 7, lower bound: -0.0161741, upper bound: 0.0165292
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 36.02
Output dim: 7, lower bound: -0.0163302, upper bound: 0.0165249
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 36.02
Output dim: 7, lower bound: -0.0163592, upper bound: 0.0165301
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 36.02
Output dim: 7, lower bound: -0.0165159, upper bound: 0.0165232

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -3.5744224, -2.7687325, -3.5743737, -2.7689767, -0.4742371, 0.4742371
1: -5.3828306, -4.0880771, -5.3827829, -4.0880651, -0.4202378, 0.4201632
2: -0.5319024, -0.3178674, -0.5318964, -0.3178715, -0.0940862, 0.0940578
3: -1.0287139, -0.6437061, -1.0287549, -0.6436560, -0.1142718, 0.1142412
4: -0.5959570, -0.0644701, -0.5960240, -0.0644743, -0.1447683, 0.1448470
5: -0.6392350, -0.2502235, -0.6390967, -0.2502253, -0.1559452, 0.1558200
6: -1.9660650, -1.2746170, -1.9660575, -1.2743568, -0.1184674, 0.1182020
7: 0.5705591, 0.9897916, 0.5717451, 0.9893701, -0.0573893, 0.0565864
8: -5.5609493, -4.4832830, -5.5608425, -4.4837961, -0.3917277, 0.3921415
9: -4.6614223, -3.6500196, -4.6613774, -3.6498318, -0.4075915, 0.4071286

Time for backsubstitution: 6.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 236
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 3526
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 3532
type: A, layer: 1, pos: 3549
type: A, layer: 1, pos: 3095
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 3499
type: A, layer: 1, pos: 2875
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 2870
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2874
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3387
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 100
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2662
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2808
type: A, layer: 1, pos: 3510
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 225
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2844
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2270
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 2959
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2945
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 2938
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2333
type: A, layer: 1, pos: 330
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2944
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 2338
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 435
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 2534
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 3209
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3366
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3374
type: A, layer: 1, pos: 3419
type: A, layer: 1, pos: 3589
type: A, layer: 1, pos: 3592
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3596
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 236

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0161735, upper bound: 0.0162992
time: 147.39 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0161737, upper bound: 0.0165247
time: 6.89 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -3.5749106, -2.7687318, -3.5749652, -2.7687316, -0.4746513, 0.4742360
1: -5.3830695, -4.0880761, -5.3830690, -4.0879431, -0.4205668, 0.4201130
2: -0.5319082, -0.3178259, -0.5319114, -0.3178242, -0.0941057, 0.0941459
3: -1.0287166, -0.6436853, -1.0287175, -0.6435368, -0.1144203, 0.1142767
4: -0.5959620, -0.0644429, -0.5960636, -0.0644413, -0.1448152, 0.1448807
5: -0.6392423, -0.2499303, -0.6392442, -0.2498785, -0.1559410, 0.1561630
6: -1.9661063, -1.2746170, -1.9661068, -1.2743422, -0.1185282, 0.1181960
7: 0.5705535, 0.9923236, 0.5702822, 0.9923234, -0.0562965, 0.0606171
8: -5.5619869, -4.4832535, -5.5620489, -4.4832420, -0.3931729, 0.3918694
9: -4.6616230, -3.6496699, -4.6616483, -3.6494393, -0.4081985, 0.4080490

Time for backsubstitution: 6.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 236
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 3526
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 3532
type: A, layer: 1, pos: 3549
type: A, layer: 1, pos: 3095
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 3499
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 2875
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 2870
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2874
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3387
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 100
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2662
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2808
type: A, layer: 1, pos: 3510
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 225
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2844
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2270
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 2959
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2945
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 2938
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2333
type: A, layer: 1, pos: 330
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2944
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 2338
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 435
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 2534
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 3209
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3366
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3374
type: A, layer: 1, pos: 3419
type: A, layer: 1, pos: 3589
type: A, layer: 1, pos: 3592
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3596
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 236

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0163293, upper bound: 0.0163067
time: 16.15 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0163293, upper bound: 0.0165228
time: 36.12 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -3.5749249, -2.7684808, -3.5746424, -2.7689760, -0.4757230, 0.4749675
1: -5.3847256, -4.0870266, -5.3827858, -4.0871444, -0.4231707, 0.4202511
2: -0.5319605, -0.3178507, -0.5319166, -0.3178568, -0.0944215, 0.0940983
3: -1.0308751, -0.6424456, -1.0287597, -0.6425893, -0.1175592, 0.1144270
4: -0.5967041, -0.0631617, -0.5966587, -0.0644689, -0.1448976, 0.1467928
5: -0.6400576, -0.2497691, -0.6391103, -0.2498280, -0.1571933, 0.1559332
6: -1.9696116, -1.2726088, -1.9660580, -1.2726239, -0.1237460, 0.1183859
7: 0.5684752, 0.9933615, 0.5700101, 0.9893763, -0.0576824, 0.0619123
8: -5.5614123, -4.4824014, -5.5612440, -4.4837179, -0.3919766, 0.3933322
9: -4.6645465, -3.6483369, -4.6615338, -3.6483834, -0.4121173, 0.4077595

Time for backsubstitution: 6.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 236
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 3526
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 3532
type: A, layer: 1, pos: 3549
type: A, layer: 1, pos: 3095
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 3499
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 2875
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 2870
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2874
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3387
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 100
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2662
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2808
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 3510
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 225
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2844
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2270
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 2959
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2945
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 2938
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2333
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 330
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2944
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 2338
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 435
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 2534
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 3209
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3366
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3374
type: A, layer: 1, pos: 3419
type: A, layer: 1, pos: 3589
type: A, layer: 1, pos: 3592
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3596
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 236

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0163587, upper bound: 0.0163054
time: 49.85 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0163584, upper bound: 0.0165259
time: 50.86 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -3.5754139, -2.7684789, -3.5752332, -2.7687309, -0.4761376, 0.4749664
1: -5.3849630, -4.0870266, -5.3830729, -4.0870214, -0.4235001, 0.4202006
2: -0.5319662, -0.3178093, -0.5319315, -0.3178096, -0.0944410, 0.0941864
3: -1.0308781, -0.6424246, -1.0287216, -0.6424701, -0.1177078, 0.1144620
4: -0.5967098, -0.0631332, -0.5966984, -0.0644357, -0.1449443, 0.1468266
5: -0.6400650, -0.2494757, -0.6392578, -0.2494812, -0.1571892, 0.1562764
6: -1.9696527, -1.2726082, -1.9661076, -1.2726089, -0.1238069, 0.1183797
7: 0.5684702, 0.9958943, 0.5685474, 0.9923293, -0.0565895, 0.0652426
8: -5.5624504, -4.4823723, -5.5624495, -4.4831634, -0.3934219, 0.3930602
9: -4.6647468, -3.6479864, -4.6618032, -3.6479902, -0.4127241, 0.4086796

Time for backsubstitution: 5.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 236
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 3526
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 3532
type: A, layer: 1, pos: 3549
type: A, layer: 1, pos: 3095
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 3499
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 2875
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 2870
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2874
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3387
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 100
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2662
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2808
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 3510
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 225
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2844
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2270
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 2959
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2945
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 2938
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2333
type: A, layer: 1, pos: 330
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2944
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 2338
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 435
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 2534
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 3209
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3366
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3374
type: A, layer: 1, pos: 3419
type: A, layer: 1, pos: 3589
type: A, layer: 1, pos: 3592
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3596
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 236

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165148, upper bound: 0.0163043
time: 78.45 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165151, upper bound: 0.0165202
time: 68.77 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 153.29 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 153.29
Output dim: 7, lower bound: -0.0161735, upper bound: 0.0162992
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 153.29
Output dim: 7, lower bound: -0.0161737, upper bound: 0.0165247
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 153.29
Output dim: 7, lower bound: -0.0163293, upper bound: 0.0163067
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 153.29
Output dim: 7, lower bound: -0.0163293, upper bound: 0.0165228
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 153.29
Output dim: 7, lower bound: -0.0163587, upper bound: 0.0163054
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 153.29
Output dim: 7, lower bound: -0.0163584, upper bound: 0.0165259
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 153.29
Output dim: 7, lower bound: -0.0165148, upper bound: 0.0163043
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 153.29
Output dim: 7, lower bound: -0.0165151, upper bound: 0.0165202

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3.5744190, -2.7687325, -3.5743718, -2.7689769, -0.4738561, 0.4742365
1: -5.3828301, -4.0880828, -5.3827825, -4.0880694, -0.4202375, 0.4152316
2: -0.5318949, -0.3178674, -0.5318907, -0.3178715, -0.0835726, 0.0940510
3: -1.0287139, -0.6437063, -1.0287551, -0.6436563, -0.1142355, 0.1142408
4: -0.5959527, -0.0644702, -0.5960208, -0.0644747, -0.1392938, 0.1448051
5: -0.6392277, -0.2502235, -0.6390910, -0.2502254, -0.1522738, 0.1558185
6: -1.9660517, -1.2746170, -1.9660473, -1.2743568, -0.1112765, 0.1181939
7: 0.5705591, 0.9897872, 0.5717451, 0.9893667, -0.0573892, 0.0505744
8: -5.5609493, -4.4832993, -5.5608425, -4.4838080, -0.3917273, 0.3823106
9: -4.6614118, -3.6500328, -4.6613698, -3.6498415, -0.4075828, 0.3889560

Time for backsubstitution: 6.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3526
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 3532
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 3095
type: B, layer: 1, pos: 236
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 3272
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2875
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 2870
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2874
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 100
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2662
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2975
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2808
type: B, layer: 1, pos: 3510
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 225
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2844
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 2270
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 2959
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 3525
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2858
type: B, layer: 1, pos: 2938
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2333
type: B, layer: 1, pos: 330
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2944
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 435
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 2534
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 3209
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3366
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3374
type: B, layer: 1, pos: 3419
type: B, layer: 1, pos: 3589
type: B, layer: 1, pos: 3592
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3596
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3437

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0160460, upper bound: 0.0165237
time: 93.21 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0161715, upper bound: 0.0165248
time: 102.08 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.5749075, -2.7687316, -3.5749629, -2.7687314, -0.4742707, 0.4742353
1: -5.3830695, -4.0880823, -5.3830690, -4.0879474, -0.4205669, 0.4151810
2: -0.5319007, -0.3178259, -0.5319057, -0.3178242, -0.0835920, 0.0941392
3: -1.0287163, -0.6436856, -1.0287173, -0.6435368, -0.1143840, 0.1142763
4: -0.5959581, -0.0644429, -0.5960605, -0.0644414, -0.1393407, 0.1448388
5: -0.6392343, -0.2499307, -0.6392381, -0.2498783, -0.1522698, 0.1561616
6: -1.9660934, -1.2746170, -1.9660968, -1.2743422, -0.1113373, 0.1181878
7: 0.5705535, 0.9923195, 0.5702822, 0.9923199, -0.0562964, 0.0546050
8: -5.5619869, -4.4832687, -5.5620489, -4.4832540, -0.3931725, 0.3820388
9: -4.6616130, -3.6496830, -4.6616402, -3.6494496, -0.4081898, 0.3898761

Time for backsubstitution: 6.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3526
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 3532
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 3095
type: B, layer: 1, pos: 236
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 3272
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2875
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 2870
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2874
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 100
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2662
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2975
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2808
type: B, layer: 1, pos: 3510
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 225
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2844
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 2270
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 2959
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 3525
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2858
type: B, layer: 1, pos: 2938
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2333
type: B, layer: 1, pos: 330
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2944
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 435
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 2534
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 3209
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3366
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3374
type: B, layer: 1, pos: 3419
type: B, layer: 1, pos: 3589
type: B, layer: 1, pos: 3592
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3596
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3437

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0162029, upper bound: 0.0165206
time: 5.04 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0163276, upper bound: 0.0165155
time: 227.35 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.5749221, -2.7684805, -3.5746403, -2.7689760, -0.4753423, 0.4749666
1: -5.3847256, -4.0870328, -5.3827863, -4.0871491, -0.4231704, 0.4153197
2: -0.5319530, -0.3178507, -0.5319110, -0.3178568, -0.0839078, 0.0940915
3: -1.0308752, -0.6424456, -1.0287592, -0.6425896, -0.1175229, 0.1144266
4: -0.5967000, -0.0631618, -0.5966555, -0.0644691, -0.1394232, 0.1467509
5: -0.6400501, -0.2497691, -0.6391045, -0.2498280, -0.1535220, 0.1559316
6: -1.9695987, -1.2726088, -1.9660485, -1.2726239, -0.1165551, 0.1183776
7: 0.5684752, 0.9933574, 0.5700100, 0.9893728, -0.0576823, 0.0559002
8: -5.5614123, -4.4824162, -5.5612440, -4.4837294, -0.3919762, 0.3835017
9: -4.6645365, -3.6483488, -4.6615257, -3.6483922, -0.4121085, 0.3895867

Time for backsubstitution: 6.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3526
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 3532
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 3095
type: B, layer: 1, pos: 236
type: B, layer: 1, pos: 3272
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2875
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 2870
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2874
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 100
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2662
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2975
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2808
type: B, layer: 1, pos: 3510
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 225
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2844
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 2270
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 2959
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 3525
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2858
type: B, layer: 1, pos: 2938
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2333
type: B, layer: 1, pos: 330
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2944
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 435
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 2534
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 3209
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3366
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3374
type: B, layer: 1, pos: 3419
type: B, layer: 1, pos: 3589
type: B, layer: 1, pos: 3592
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3596
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3437

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0162315, upper bound: 0.0165225
time: 106.60 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0163563, upper bound: 0.0165226
time: 116.48 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.5744991, -2.7686129, -3.5745313, -2.7687316, -0.4751935, 0.4740944
1: -5.3841605, -4.0913315, -5.3830671, -4.0904384, -0.4192256, 0.4158599
2: -0.5226382, -0.3195500, -0.5245213, -0.3178098, -0.0851811, 0.0851633
3: -1.0305758, -0.6425567, -1.0284883, -0.6425112, -0.1173880, 0.1141143
4: -0.5916364, -0.0640614, -0.5926759, -0.0644360, -0.1399330, 0.1417534
5: -0.6361965, -0.2502289, -0.6361960, -0.2495221, -0.1532487, 0.1523999
6: -1.9633652, -1.2737323, -1.9611137, -1.2726085, -0.1175022, 0.1121631
7: 0.5694950, 0.9906470, 0.5685495, 0.9881628, -0.0513568, 0.0599735
8: -5.5606403, -4.4922018, -5.5624490, -4.4909067, -0.3836828, 0.3831269
9: -4.6615767, -3.6646886, -4.6616683, -3.6613183, -0.3967904, 0.3923557

Time for backsubstitution: 6.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3526
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 3532
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 3095
type: B, layer: 1, pos: 3272
type: B, layer: 1, pos: 236
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2875
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 2870
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2874
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 100
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2662
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2975
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2808
type: B, layer: 1, pos: 3510
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 225
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2844
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 2270
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 2959
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 3525
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2858
type: B, layer: 1, pos: 2938
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2333
type: B, layer: 1, pos: 330
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2944
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 435
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 2534
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 3209
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3366
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3374
type: B, layer: 1, pos: 3419
type: B, layer: 1, pos: 3589
type: B, layer: 1, pos: 3592
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3596
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3437

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0163884, upper bound: 0.0163052
time: 63.56 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165136, upper bound: 0.0162978
time: 5.26 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.5754106, -2.7684793, -3.5752311, -2.7687304, -0.4757571, 0.4749655
1: -5.3849630, -4.0870323, -5.3830738, -4.0870261, -0.4234998, 0.4152691
2: -0.5319587, -0.3178093, -0.5319257, -0.3178096, -0.0839273, 0.0941796
3: -1.0308774, -0.6424246, -1.0287215, -0.6424696, -0.1176713, 0.1144617
4: -0.5967058, -0.0631332, -0.5966952, -0.0644357, -0.1394701, 0.1467848
5: -0.6400571, -0.2494757, -0.6392518, -0.2494812, -0.1535179, 0.1562749
6: -1.9696393, -1.2726080, -1.9660972, -1.2726089, -0.1166160, 0.1183716
7: 0.5684702, 0.9958895, 0.5685474, 0.9923261, -0.0565895, 0.0592136
8: -5.5624504, -4.4823871, -5.5624495, -4.4831748, -0.3934216, 0.3832294
9: -4.6647367, -3.6479998, -4.6617947, -3.6480005, -0.4127151, 0.3905065

Time for backsubstitution: 6.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3437
type: B, layer: 1, pos: 3526
type: B, layer: 1, pos: 3543
type: B, layer: 1, pos: 602
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 566
type: B, layer: 1, pos: 3532
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 3095
type: B, layer: 1, pos: 236
type: B, layer: 1, pos: 3272
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 3287
type: B, layer: 1, pos: 2875
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 2870
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2874
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 3288
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2203
type: B, layer: 1, pos: 2517
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 100
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 2662
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2498
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2975
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2808
type: B, layer: 1, pos: 3510
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 2559
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 225
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 51
type: B, layer: 1, pos: 2844
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 2949
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 2270
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 2959
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 770
type: B, layer: 1, pos: 3525
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2858
type: B, layer: 1, pos: 2938
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2333
type: B, layer: 1, pos: 330
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2349
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 2944
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 3013
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 3305
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 88
type: B, layer: 1, pos: 59
type: B, layer: 1, pos: 435
type: B, layer: 1, pos: 584
type: B, layer: 1, pos: 660
type: B, layer: 1, pos: 661
type: B, layer: 1, pos: 779
type: B, layer: 1, pos: 824
type: B, layer: 1, pos: 2534
type: B, layer: 1, pos: 2684
type: B, layer: 1, pos: 2692
type: B, layer: 1, pos: 3209
type: B, layer: 1, pos: 3363
type: B, layer: 1, pos: 3366
type: B, layer: 1, pos: 3368
type: B, layer: 1, pos: 3374
type: B, layer: 1, pos: 3419
type: B, layer: 1, pos: 3589
type: B, layer: 1, pos: 3592
type: B, layer: 1, pos: 3594
type: B, layer: 1, pos: 3595
type: B, layer: 1, pos: 3596
type: B, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 3437

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0163886, upper bound: 0.0165180
time: 68.14 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165134, upper bound: 0.0165260
time: 60.43 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 135.04 seconds
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 135.04
Output dim: 7, lower bound: -0.0160460, upper bound: 0.0165237
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 135.04
Output dim: 7, lower bound: -0.0161715, upper bound: 0.0165248
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 135.04
Output dim: 7, lower bound: -0.0162029, upper bound: 0.0165206
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 135.04
Output dim: 7, lower bound: -0.0163276, upper bound: 0.0165155
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 135.04
Output dim: 7, lower bound: -0.0162315, upper bound: 0.0165225
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 135.04
Output dim: 7, lower bound: -0.0163563, upper bound: 0.0165226
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 135.04
Output dim: 7, lower bound: -0.0163884, upper bound: 0.0163052
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 135.04
Output dim: 7, lower bound: -0.0165136, upper bound: 0.0162978
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 135.04
Output dim: 7, lower bound: -0.0163886, upper bound: 0.0165180
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 135.04
Output dim: 7, lower bound: -0.0165134, upper bound: 0.0165260

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3.5730634, -2.7687342, -3.5724690, -2.7699211, -0.4713333, 0.4723947
1: -5.3828259, -4.0880866, -5.3827724, -4.0880651, -0.4202064, 0.4152052
2: -0.5311634, -0.3178679, -0.5310502, -0.3186400, -0.0821985, 0.0932766
3: -1.0287079, -0.6442759, -1.0282631, -0.6444852, -0.1135495, 0.1135439
4: -0.5953404, -0.0644706, -0.5952268, -0.0649334, -0.1388103, 0.1440845
5: -0.6390961, -0.2503148, -0.6389173, -0.2504704, -0.1516616, 0.1555752
6: -1.9655700, -1.2746179, -1.9654927, -1.2747964, -0.1107306, 0.1178389
7: 0.5705625, 0.9883873, 0.5735471, 0.9877663, -0.0558003, 0.0473286
8: -5.5609450, -4.4836464, -5.5604687, -4.4842863, -0.3913016, 0.3816392
9: -4.6612568, -3.6506009, -4.6607275, -3.6505089, -0.4069360, 0.3880076

Time for backsubstitution: 6.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3526
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 3532
type: A, layer: 1, pos: 3549
type: A, layer: 1, pos: 3095
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 3499
type: A, layer: 1, pos: 2875
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 2870
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2874
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3387
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 100
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2662
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2808
type: A, layer: 1, pos: 3510
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 225
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2844
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2270
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 2959
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2945
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 2938
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2333
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 330
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2944
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 2338
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 435
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 2534
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 3209
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3366
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3374
type: A, layer: 1, pos: 3419
type: A, layer: 1, pos: 3589
type: A, layer: 1, pos: 3592
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3596
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3526

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0159387, upper bound: 0.0165237
time: 35.81 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0160455, upper bound: 0.0165256
time: 61.22 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3.5743709, -2.7687328, -3.5743167, -2.7689772, -0.4738431, 0.4737756
1: -5.3828220, -4.0880828, -5.3827720, -4.0880694, -0.4202236, 0.4152108
2: -0.5318932, -0.3178674, -0.5318887, -0.3178715, -0.0835574, 0.0926560
3: -1.0287139, -0.6437280, -1.0287549, -0.6436785, -0.1137667, 0.1139891
4: -0.5959456, -0.0644704, -0.5960124, -0.0644747, -0.1386051, 0.1448791
5: -0.6392210, -0.2502238, -0.6390833, -0.2502255, -0.1524150, 0.1556090
6: -1.9660480, -1.2746172, -1.9660437, -1.2743574, -0.1111025, 0.1178295
7: 0.5705589, 0.9897784, 0.5717450, 0.9893567, -0.0544440, 0.0505687
8: -5.5609493, -4.4833169, -5.5608425, -4.4838281, -0.3914354, 0.3822780
9: -4.6614056, -3.6500349, -4.6613641, -3.6498439, -0.4071497, 0.3889148

Time for backsubstitution: 6.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3526
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 3532
type: A, layer: 1, pos: 3549
type: A, layer: 1, pos: 3095
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 3499
type: A, layer: 1, pos: 2875
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 2870
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2874
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3387
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 100
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2662
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2808
type: A, layer: 1, pos: 3510
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 225
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2844
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2270
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 2959
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2945
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 2938
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2333
type: A, layer: 1, pos: 330
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2944
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 2338
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 435
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 2534
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 3209
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3366
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3374
type: A, layer: 1, pos: 3419
type: A, layer: 1, pos: 3589
type: A, layer: 1, pos: 3592
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3596
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3526

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0160634, upper bound: 0.0165306
time: 33.92 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0161709, upper bound: 0.0165236
time: 10.54 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3.5735531, -2.7687328, -3.5730574, -2.7696748, -0.4717474, 0.4723918
1: -5.3830652, -4.0880852, -5.3830590, -4.0879426, -0.4205357, 0.4151546
2: -0.5311689, -0.3178264, -0.5310646, -0.3185928, -0.0822180, 0.0933644
3: -1.0287106, -0.6442562, -1.0282255, -0.6444558, -0.1135745, 0.1135794
4: -0.5953454, -0.0644430, -0.5952475, -0.0649006, -0.1388575, 0.1440942
5: -0.6391028, -0.2500218, -0.6390649, -0.2501243, -0.1516561, 0.1559181
6: -1.9656112, -1.2746180, -1.9655417, -1.2747809, -0.1107915, 0.1178324
7: 0.5705575, 0.9909191, 0.5723066, 0.9907190, -0.0547074, 0.0511022
8: -5.5619822, -4.4836168, -5.5616741, -4.4837317, -0.3927467, 0.3813659
9: -4.6614594, -3.6502507, -4.6610007, -3.6501188, -0.4075413, 0.3889288

Time for backsubstitution: 6.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3526
type: A, layer: 1, pos: 3543
type: A, layer: 1, pos: 602
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 3532
type: A, layer: 1, pos: 3549
type: A, layer: 1, pos: 3095
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 3499
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 2875
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 2870
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 2158
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2867
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2874
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3387
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2203
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 100
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 99
type: A, layer: 1, pos: 2662
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 2275
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2531
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2808
type: A, layer: 1, pos: 3510
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 2062
type: A, layer: 1, pos: 2951
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 776
type: A, layer: 1, pos: 225
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 2844
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2270
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 2959
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2945
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 2938
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2333
type: A, layer: 1, pos: 330
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 2053
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2944
type: A, layer: 1, pos: 684
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 2338
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 685
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 111
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 3013
type: A, layer: 1, pos: 748
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 649
type: A, layer: 1, pos: 756
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 787
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 59
type: A, layer: 1, pos: 435
type: A, layer: 1, pos: 584
type: A, layer: 1, pos: 660
type: A, layer: 1, pos: 661
type: A, layer: 1, pos: 779
type: A, layer: 1, pos: 824
type: A, layer: 1, pos: 2534
type: A, layer: 1, pos: 2684
type: A, layer: 1, pos: 2692
type: A, layer: 1, pos: 3209
type: A, layer: 1, pos: 3363
type: A, layer: 1, pos: 3366
type: A, layer: 1, pos: 3368
type: A, layer: 1, pos: 3374
type: A, layer: 1, pos: 3419
type: A, layer: 1, pos: 3589
type: A, layer: 1, pos: 3592
type: A, layer: 1, pos: 3594
type: A, layer: 1, pos: 3595
type: A, layer: 1, pos: 3596
type: A, layer: 1, pos: 3597

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3526

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0160945, upper bound: 0.0165239
time: 66.45 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0162018, upper bound: 0.0165288
time: 53.22 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 33.81 + 1782.54 = 1816.35 seconds
