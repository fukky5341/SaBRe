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
execution time: IAR + RelationalAnalysis = 7.81 + 25.98 = 33.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0165178, upper bound: 0.0165269

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3527
type: B, layer: 1, pos: 3527
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 3499
type: B, layer: 1, pos: 3499
type: A, layer: 1, pos: 236
type: B, layer: 1, pos: 236
type: A, layer: 1, pos: 3526
type: B, layer: 1, pos: 3526
type: A, layer: 1, pos: 3543
type: B, layer: 1, pos: 3543
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 3095
type: B, layer: 1, pos: 3095
type: A, layer: 1, pos: 3532
type: B, layer: 1, pos: 3532
type: A, layer: 1, pos: 3549
type: B, layer: 1, pos: 3549
type: A, layer: 1, pos: 3272
type: B, layer: 1, pos: 3272
type: A, layer: 1, pos: 2650
type: B, layer: 1, pos: 2650
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 2868
type: B, layer: 1, pos: 2868
type: A, layer: 1, pos: 3287
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 2869
type: B, layer: 1, pos: 2869
type: A, layer: 1, pos: 2870
type: B, layer: 1, pos: 2870
type: A, layer: 1, pos: 2875
type: B, layer: 1, pos: 2875
type: A, layer: 1, pos: 2867
type: B, layer: 1, pos: 2867
type: A, layer: 1, pos: 2158
type: B, layer: 1, pos: 2158
type: A, layer: 1, pos: 2874
type: B, layer: 1, pos: 2874
type: A, layer: 1, pos: 2188
type: B, layer: 1, pos: 2188
type: A, layer: 1, pos: 2230
type: B, layer: 1, pos: 2230
type: A, layer: 1, pos: 2203
type: B, layer: 1, pos: 2203
type: A, layer: 1, pos: 2577
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2866
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 3100
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2228
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 2662
type: B, layer: 1, pos: 2662
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 3289
type: B, layer: 1, pos: 3289
type: A, layer: 1, pos: 2532
type: B, layer: 1, pos: 2532
type: A, layer: 1, pos: 2422
type: B, layer: 1, pos: 2422
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 2633
type: A, layer: 1, pos: 2513
type: B, layer: 1, pos: 2513
type: A, layer: 1, pos: 2517
type: B, layer: 1, pos: 2517
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2303
type: B, layer: 1, pos: 2303
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 100
type: B, layer: 1, pos: 100
type: A, layer: 1, pos: 2499
type: B, layer: 1, pos: 2499
type: A, layer: 1, pos: 2484
type: B, layer: 1, pos: 2484
type: A, layer: 1, pos: 2301
type: B, layer: 1, pos: 2301
type: A, layer: 1, pos: 2302
type: B, layer: 1, pos: 2302
type: A, layer: 1, pos: 2873
type: B, layer: 1, pos: 2873
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 2604
type: B, layer: 1, pos: 2604
type: A, layer: 1, pos: 2975
type: B, layer: 1, pos: 2975
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3480
type: B, layer: 1, pos: 3480
type: A, layer: 1, pos: 2498
type: B, layer: 1, pos: 2498
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2062
type: B, layer: 1, pos: 2062
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 2087
type: B, layer: 1, pos: 2087
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2951
type: A, layer: 1, pos: 3510
type: B, layer: 1, pos: 3510
type: B, layer: 1, pos: 2073
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 2980
type: B, layer: 1, pos: 2980
type: A, layer: 1, pos: 2808
type: B, layer: 1, pos: 2808
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 225
type: B, layer: 1, pos: 225
type: A, layer: 1, pos: 2559
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 2949
type: B, layer: 1, pos: 2949
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2483
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2270
type: B, layer: 1, pos: 2270
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2057
type: B, layer: 1, pos: 2057
type: A, layer: 1, pos: 2977
type: B, layer: 1, pos: 2977
type: A, layer: 1, pos: 3290
type: B, layer: 1, pos: 3290
type: A, layer: 1, pos: 2844
type: B, layer: 1, pos: 2844
type: A, layer: 1, pos: 2959
type: B, layer: 1, pos: 2959
type: A, layer: 1, pos: 2945
type: B, layer: 1, pos: 2945
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 2976
type: B, layer: 1, pos: 2976
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 3525
type: B, layer: 1, pos: 3525
type: A, layer: 1, pos: 2938
type: B, layer: 1, pos: 2938
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2944
type: B, layer: 1, pos: 2944
type: A, layer: 1, pos: 2333
type: B, layer: 1, pos: 2333
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2547
type: B, layer: 1, pos: 2547
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 2053
type: B, layer: 1, pos: 2053
type: A, layer: 1, pos: 2858
type: B, layer: 1, pos: 2858
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2338
type: B, layer: 1, pos: 2338
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 3008
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 3354
type: B, layer: 1, pos: 3354
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 2213
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 2648
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2106
type: B, layer: 1, pos: 2106
type: A, layer: 1, pos: 2479
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 330
type: A, layer: 1, pos: 330
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 3305
type: B, layer: 1, pos: 3305
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 3013
type: B, layer: 1, pos: 3013
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 2263
type: B, layer: 1, pos: 2263
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 2232
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 88
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
type: A, layer: 1, pos: 3527

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0163313, upper bound: 0.0165320
time: 6.55 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165165, upper bound: 0.0165313
time: 47.88 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 54.51 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 54.51
Output dim: 7, lower bound: -0.0163313, upper bound: 0.0165320
NS_A2, status: Status.UNKNOWN, split count: 1, time: 54.51
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

Time for backsubstitution: 5.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3499
type: A, layer: 1, pos: 3499
type: B, layer: 1, pos: 236
type: A, layer: 1, pos: 236
type: B, layer: 1, pos: 3526
type: A, layer: 1, pos: 3526
type: B, layer: 1, pos: 3543
type: A, layer: 1, pos: 3543
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 3095
type: B, layer: 1, pos: 3095
type: A, layer: 1, pos: 3532
type: B, layer: 1, pos: 3532
type: B, layer: 1, pos: 3549
type: A, layer: 1, pos: 3549
type: A, layer: 1, pos: 3272
type: B, layer: 1, pos: 3272
type: A, layer: 1, pos: 2650
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 2868
type: A, layer: 1, pos: 2868
type: A, layer: 1, pos: 3287
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 2869
type: B, layer: 1, pos: 2869
type: B, layer: 1, pos: 2870
type: A, layer: 1, pos: 2870
type: B, layer: 1, pos: 2875
type: A, layer: 1, pos: 2875
type: B, layer: 1, pos: 2867
type: A, layer: 1, pos: 2867
type: B, layer: 1, pos: 2158
type: A, layer: 1, pos: 2158
type: B, layer: 1, pos: 3527
type: A, layer: 1, pos: 2874
type: B, layer: 1, pos: 2874
type: B, layer: 1, pos: 2188
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2230
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2203
type: A, layer: 1, pos: 2203
type: B, layer: 1, pos: 2577
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2866
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2228
type: B, layer: 1, pos: 2228
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 2662
type: A, layer: 1, pos: 2662
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 3387
type: A, layer: 1, pos: 3289
type: B, layer: 1, pos: 3289
type: A, layer: 1, pos: 2532
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 2422
type: A, layer: 1, pos: 2422
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2633
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 2513
type: A, layer: 1, pos: 2513
type: B, layer: 1, pos: 2517
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 2303
type: A, layer: 1, pos: 2303
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 100
type: A, layer: 1, pos: 100
type: B, layer: 1, pos: 2499
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2484
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 2301
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2302
type: B, layer: 1, pos: 2302
type: A, layer: 1, pos: 2873
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2604
type: B, layer: 1, pos: 2604
type: A, layer: 1, pos: 2975
type: B, layer: 1, pos: 2975
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3480
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 2498
type: B, layer: 1, pos: 2498
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2062
type: A, layer: 1, pos: 2062
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 2087
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 3510
type: A, layer: 1, pos: 3510
type: B, layer: 1, pos: 2073
type: A, layer: 1, pos: 2073
type: B, layer: 1, pos: 2980
type: A, layer: 1, pos: 2980
type: B, layer: 1, pos: 2808
type: A, layer: 1, pos: 2808
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 225
type: B, layer: 1, pos: 225
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 2949
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 2483
type: A, layer: 1, pos: 2483
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 2270
type: A, layer: 1, pos: 2270
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2057
type: A, layer: 1, pos: 2057
type: B, layer: 1, pos: 2977
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 3290
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2844
type: A, layer: 1, pos: 2844
type: B, layer: 1, pos: 2959
type: A, layer: 1, pos: 2959
type: B, layer: 1, pos: 2945
type: A, layer: 1, pos: 2945
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 2976
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 3525
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 2938
type: B, layer: 1, pos: 2938
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2944
type: A, layer: 1, pos: 2944
type: B, layer: 1, pos: 2333
type: A, layer: 1, pos: 2333
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 2547
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 2053
type: B, layer: 1, pos: 2053
type: A, layer: 1, pos: 2858
type: B, layer: 1, pos: 2858
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2338
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 3008
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 3354
type: B, layer: 1, pos: 3354
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2213
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2648
type: B, layer: 1, pos: 2648
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2106
type: B, layer: 1, pos: 2106
type: A, layer: 1, pos: 2479
type: B, layer: 1, pos: 2479
type: A, layer: 1, pos: 330
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 330
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 3305
type: A, layer: 1, pos: 3305
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 3013
type: B, layer: 1, pos: 3013
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 2263
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 2232
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 88
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

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0162046, upper bound: 0.0165284
time: 6.00 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0163297, upper bound: 0.0165240
time: 64.60 seconds

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

Time for backsubstitution: 6.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3437
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3499
type: A, layer: 1, pos: 3499
type: B, layer: 1, pos: 236
type: A, layer: 1, pos: 236
type: B, layer: 1, pos: 3526
type: A, layer: 1, pos: 3526
type: B, layer: 1, pos: 3543
type: A, layer: 1, pos: 3543
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 3542
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 3095
type: B, layer: 1, pos: 3095
type: A, layer: 1, pos: 3532
type: B, layer: 1, pos: 3532
type: B, layer: 1, pos: 3549
type: A, layer: 1, pos: 3549
type: A, layer: 1, pos: 3272
type: B, layer: 1, pos: 3272
type: A, layer: 1, pos: 2650
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 2868
type: A, layer: 1, pos: 2868
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 3287
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 2869
type: B, layer: 1, pos: 2869
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 2870
type: A, layer: 1, pos: 2870
type: B, layer: 1, pos: 2875
type: A, layer: 1, pos: 2875
type: B, layer: 1, pos: 2867
type: A, layer: 1, pos: 2867
type: B, layer: 1, pos: 2158
type: A, layer: 1, pos: 2158
type: B, layer: 1, pos: 3527
type: A, layer: 1, pos: 2874
type: B, layer: 1, pos: 2874
type: B, layer: 1, pos: 2188
type: A, layer: 1, pos: 2188
type: A, layer: 1, pos: 2230
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2203
type: A, layer: 1, pos: 2203
type: B, layer: 1, pos: 2577
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 2866
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2662
type: A, layer: 1, pos: 2662
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 3387
type: A, layer: 1, pos: 3289
type: B, layer: 1, pos: 3289
type: A, layer: 1, pos: 2532
type: B, layer: 1, pos: 2532
type: B, layer: 1, pos: 2422
type: A, layer: 1, pos: 2422
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 2142
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 3227
type: B, layer: 1, pos: 2633
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 2513
type: A, layer: 1, pos: 2513
type: B, layer: 1, pos: 2517
type: A, layer: 1, pos: 2517
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 2304
type: B, layer: 1, pos: 2303
type: A, layer: 1, pos: 2303
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 100
type: A, layer: 1, pos: 100
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 2499
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2484
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 2301
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2302
type: B, layer: 1, pos: 2302
type: A, layer: 1, pos: 2873
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2604
type: B, layer: 1, pos: 2604
type: A, layer: 1, pos: 2975
type: B, layer: 1, pos: 2975
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3480
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 2498
type: B, layer: 1, pos: 2498
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 2062
type: A, layer: 1, pos: 2062
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 2087
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 2073
type: A, layer: 1, pos: 2073
type: B, layer: 1, pos: 3510
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 2808
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 3510
type: A, layer: 1, pos: 2808
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 225
type: B, layer: 1, pos: 225
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 2949
type: B, layer: 1, pos: 2949
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 2483
type: A, layer: 1, pos: 2483
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2270
type: A, layer: 1, pos: 2270
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2057
type: B, layer: 1, pos: 2977
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 3290
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2844
type: A, layer: 1, pos: 2844
type: B, layer: 1, pos: 2959
type: A, layer: 1, pos: 2959
type: B, layer: 1, pos: 2945
type: A, layer: 1, pos: 2945
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 2976
type: B, layer: 1, pos: 2976
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 699
type: A, layer: 1, pos: 2938
type: B, layer: 1, pos: 2938
type: B, layer: 1, pos: 3525
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2944
type: A, layer: 1, pos: 2944
type: A, layer: 1, pos: 3525
type: B, layer: 1, pos: 2333
type: A, layer: 1, pos: 2333
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: A, layer: 1, pos: 2858
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 2547
type: A, layer: 1, pos: 2547
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 2053
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2858
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2338
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 3008
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 3354
type: B, layer: 1, pos: 3354
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 330
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2213
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2106
type: B, layer: 1, pos: 2106
type: A, layer: 1, pos: 2479
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 2648
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 3305
type: A, layer: 1, pos: 3305
type: B, layer: 1, pos: 330
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 3013
type: B, layer: 1, pos: 3013
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 2263
type: B, layer: 1, pos: 2263
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 88
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
type: B, layer: 1, pos: 2232

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3437

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0163901, upper bound: 0.0165276
time: 65.67 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165155, upper bound: 0.0165283
time: 26.59 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 98.63 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 98.63
Output dim: 7, lower bound: -0.0162046, upper bound: 0.0165284
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 98.63
Output dim: 7, lower bound: -0.0163297, upper bound: 0.0165240
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 98.63
Output dim: 7, lower bound: -0.0163901, upper bound: 0.0165276
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 98.63
Output dim: 7, lower bound: -0.0165155, upper bound: 0.0165283

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -3.5735598, -2.7687325, -3.5730648, -2.7696750, -0.4721549, 0.4728420
1: -5.3830667, -4.0880795, -5.3830605, -4.0879383, -0.4205419, 0.4203922
2: -0.5311763, -0.3178229, -0.5310703, -0.3185887, -0.0927804, 0.0933711
3: -1.0287111, -0.6442438, -1.0282254, -0.6444402, -0.1136467, 0.1135884
4: -0.5953497, -0.0644412, -0.5952510, -0.0648993, -0.1443149, 0.1441570
5: -0.6391109, -0.2500184, -0.6390709, -0.2501197, -0.1556250, 0.1559346
6: -1.9656259, -1.2746180, -1.9655527, -1.2747809, -0.1179821, 0.1178980
7: 0.5705576, 0.9909303, 0.5723066, 0.9907308, -0.0587560, 0.0571208
8: -5.5619893, -4.4836025, -5.5616822, -4.4837208, -0.3927734, 0.3925793
9: -4.6614690, -3.6502171, -4.6610093, -3.6500840, -0.4075717, 0.4070427

Time for backsubstitution: 6.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3499
type: B, layer: 1, pos: 3499
type: A, layer: 1, pos: 236
type: B, layer: 1, pos: 236
type: B, layer: 1, pos: 3526
type: A, layer: 1, pos: 3526
type: B, layer: 1, pos: 3543
type: A, layer: 1, pos: 3543
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 3095
type: B, layer: 1, pos: 3095
type: A, layer: 1, pos: 3532
type: B, layer: 1, pos: 3532
type: B, layer: 1, pos: 3272
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 3549
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 2869
type: B, layer: 1, pos: 2869
type: A, layer: 1, pos: 2875
type: B, layer: 1, pos: 2875
type: A, layer: 1, pos: 2870
type: B, layer: 1, pos: 2870
type: A, layer: 1, pos: 2650
type: B, layer: 1, pos: 2650
type: A, layer: 1, pos: 2868
type: B, layer: 1, pos: 2868
type: A, layer: 1, pos: 2158
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 3527
type: A, layer: 1, pos: 2867
type: B, layer: 1, pos: 2867
type: A, layer: 1, pos: 2188
type: B, layer: 1, pos: 2188
type: A, layer: 1, pos: 2874
type: B, layer: 1, pos: 2874
type: B, layer: 1, pos: 2577
type: A, layer: 1, pos: 2577
type: A, layer: 1, pos: 3289
type: B, layer: 1, pos: 3289
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 3100
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 2532
type: B, layer: 1, pos: 2532
type: A, layer: 1, pos: 2228
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: B, layer: 1, pos: 2513
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 2203
type: B, layer: 1, pos: 2203
type: A, layer: 1, pos: 2517
type: B, layer: 1, pos: 2517
type: A, layer: 1, pos: 2230
type: B, layer: 1, pos: 2230
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2866
type: B, layer: 1, pos: 2866
type: A, layer: 1, pos: 100
type: B, layer: 1, pos: 100
type: A, layer: 1, pos: 2303
type: B, layer: 1, pos: 2303
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 2422
type: B, layer: 1, pos: 2499
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2484
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 2422
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 2301
type: B, layer: 1, pos: 2301
type: A, layer: 1, pos: 2302
type: B, layer: 1, pos: 2302
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2662
type: B, layer: 1, pos: 2662
type: A, layer: 1, pos: 2604
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 3480
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 2873
type: B, layer: 1, pos: 2873
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2498
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2975
type: B, layer: 1, pos: 2975
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 3510
type: A, layer: 1, pos: 3510
type: A, layer: 1, pos: 2062
type: B, layer: 1, pos: 2062
type: A, layer: 1, pos: 2808
type: B, layer: 1, pos: 2808
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2951
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 3290
type: A, layer: 1, pos: 3290
type: B, layer: 1, pos: 2980
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 2559
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 225
type: B, layer: 1, pos: 225
type: B, layer: 1, pos: 2949
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 2483
type: B, layer: 1, pos: 2483
type: A, layer: 1, pos: 2844
type: B, layer: 1, pos: 2844
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2087
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2977
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2270
type: B, layer: 1, pos: 2270
type: A, layer: 1, pos: 2057
type: B, layer: 1, pos: 2057
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2073
type: B, layer: 1, pos: 2073
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 2959
type: B, layer: 1, pos: 2959
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 2976
type: B, layer: 1, pos: 2976
type: A, layer: 1, pos: 2945
type: B, layer: 1, pos: 2945
type: A, layer: 1, pos: 3525
type: B, layer: 1, pos: 3525
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 2938
type: B, layer: 1, pos: 2938
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 2053
type: B, layer: 1, pos: 2053
type: A, layer: 1, pos: 2333
type: B, layer: 1, pos: 2333
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 2944
type: B, layer: 1, pos: 2944
type: A, layer: 1, pos: 2858
type: B, layer: 1, pos: 2858
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 2648
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 3354
type: B, layer: 1, pos: 3354
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 3008
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 3025
type: B, layer: 1, pos: 330
type: A, layer: 1, pos: 2338
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 2106
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 2479
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 330
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 2213
type: B, layer: 1, pos: 2213
type: A, layer: 1, pos: 3013
type: B, layer: 1, pos: 3013
type: A, layer: 1, pos: 2547
type: B, layer: 1, pos: 2547
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 2633
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 656
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 3305
type: B, layer: 1, pos: 3305
type: A, layer: 1, pos: 2263
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 2263
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 2232
type: A, layer: 1, pos: 2232
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
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

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3499

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0162035, upper bound: 0.0163651
time: 38.64 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0162036, upper bound: 0.0165234
time: 103.13 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -3.5748668, -2.7687314, -3.5749149, -2.7687316, -0.4746646, 0.4742235
1: -5.3830619, -4.0880761, -5.3830605, -4.0879426, -0.4205590, 0.4203974
2: -0.5319062, -0.3178224, -0.5319093, -0.3178201, -0.0941394, 0.0927509
3: -1.0287166, -0.6436944, -1.0287172, -0.6435448, -0.1139853, 0.1140338
4: -0.5959549, -0.0644416, -0.5960552, -0.0644398, -0.1441125, 0.1449753
5: -0.6392361, -0.2499270, -0.6392368, -0.2498742, -0.1563796, 0.1559685
6: -1.9661043, -1.2746170, -1.9661047, -1.2743422, -0.1183547, 0.1178889
7: 0.5705535, 0.9923218, 0.5702824, 0.9923216, -0.0573998, 0.0606182
8: -5.5619941, -4.4832716, -5.5620580, -4.4832630, -0.3929074, 0.3932195
9: -4.6616173, -3.6496501, -4.6616421, -3.6494160, -0.4077874, 0.4079492

Time for backsubstitution: 6.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3499
type: B, layer: 1, pos: 3499
type: A, layer: 1, pos: 236
type: B, layer: 1, pos: 236
type: A, layer: 1, pos: 3526
type: B, layer: 1, pos: 3526
type: A, layer: 1, pos: 3543
type: B, layer: 1, pos: 3543
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 3095
type: A, layer: 1, pos: 3095
type: A, layer: 1, pos: 3532
type: B, layer: 1, pos: 3532
type: A, layer: 1, pos: 3549
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 3272
type: A, layer: 1, pos: 3272
type: B, layer: 1, pos: 2650
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 2868
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 3287
type: B, layer: 1, pos: 2869
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 2870
type: B, layer: 1, pos: 2870
type: A, layer: 1, pos: 2875
type: B, layer: 1, pos: 2875
type: A, layer: 1, pos: 2867
type: B, layer: 1, pos: 2867
type: A, layer: 1, pos: 2158
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 2874
type: A, layer: 1, pos: 2874
type: A, layer: 1, pos: 2188
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2230
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2203
type: B, layer: 1, pos: 2203
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2577
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2866
type: A, layer: 1, pos: 2866
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 2662
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 2662
type: A, layer: 1, pos: 2228
type: B, layer: 1, pos: 3289
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 2422
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2532
type: A, layer: 1, pos: 2532
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 2422
type: A, layer: 1, pos: 2513
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2517
type: A, layer: 1, pos: 2517
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 2303
type: B, layer: 1, pos: 2303
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: A, layer: 1, pos: 100
type: B, layer: 1, pos: 100
type: B, layer: 1, pos: 2499
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2484
type: B, layer: 1, pos: 2484
type: A, layer: 1, pos: 2301
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 2302
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2604
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2873
type: B, layer: 1, pos: 2975
type: A, layer: 1, pos: 2975
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3480
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 2498
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2062
type: B, layer: 1, pos: 2062
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 2087
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 3510
type: B, layer: 1, pos: 3510
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 2808
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 2808
type: B, layer: 1, pos: 225
type: A, layer: 1, pos: 225
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2559
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 51
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2949
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 2483
type: B, layer: 1, pos: 2483
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2270
type: B, layer: 1, pos: 2270
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2057
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2844
type: B, layer: 1, pos: 3290
type: A, layer: 1, pos: 3290
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 2844
type: A, layer: 1, pos: 2959
type: A, layer: 1, pos: 2945
type: B, layer: 1, pos: 2959
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 2858
type: A, layer: 1, pos: 2213
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 2976
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 2976
type: B, layer: 1, pos: 2648
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 3525
type: B, layer: 1, pos: 3525
type: A, layer: 1, pos: 330
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2938
type: A, layer: 1, pos: 2938
type: A, layer: 1, pos: 2944
type: B, layer: 1, pos: 2944
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2333
type: B, layer: 1, pos: 2333
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 2547
type: B, layer: 1, pos: 2547
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 2232
type: A, layer: 1, pos: 2053
type: B, layer: 1, pos: 2053
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2338
type: B, layer: 1, pos: 2338
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 3354
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 2106
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 2479
type: B, layer: 1, pos: 3305
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 3013
type: B, layer: 1, pos: 3013
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: B, layer: 1, pos: 2263
type: A, layer: 1, pos: 2263
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 88
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
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2648
type: B, layer: 1, pos: 330
type: B, layer: 1, pos: 2213
type: A, layer: 1, pos: 2232

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3499

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0163287, upper bound: 0.0163654
time: 92.50 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0163286, upper bound: 0.0165245
time: 108.78 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -3.5740652, -2.7684796, -3.5733321, -2.7696743, -0.4736413, 0.4735714
1: -5.3849611, -4.0870299, -5.3830652, -4.0870171, -0.4234749, 0.4204798
2: -0.5312346, -0.3178062, -0.5310908, -0.3185740, -0.0931157, 0.0934115
3: -1.0308714, -0.6429822, -1.0282300, -0.6433735, -0.1169342, 0.1137745
4: -0.5960969, -0.0631330, -0.5958856, -0.0648930, -0.1444442, 0.1461029
5: -0.6399332, -0.2495629, -0.6390849, -0.2497227, -0.1568731, 0.1560477
6: -1.9691719, -1.2726095, -1.9655536, -1.2730480, -0.1232606, 0.1180817
7: 0.5684741, 0.9945004, 0.5705718, 0.9907370, -0.0590491, 0.0624469
8: -5.5624528, -4.4827213, -5.5620828, -4.4836426, -0.3930224, 0.3937700
9: -4.6645937, -3.6485326, -4.6611657, -3.6486340, -0.4120976, 0.4076737

Time for backsubstitution: 6.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3499
type: B, layer: 1, pos: 3499
type: A, layer: 1, pos: 236
type: B, layer: 1, pos: 236
type: B, layer: 1, pos: 3526
type: A, layer: 1, pos: 3526
type: B, layer: 1, pos: 3543
type: A, layer: 1, pos: 3543
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 3095
type: B, layer: 1, pos: 3095
type: A, layer: 1, pos: 3532
type: B, layer: 1, pos: 3532
type: B, layer: 1, pos: 3272
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 3549
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 3287
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 2869
type: B, layer: 1, pos: 2869
type: A, layer: 1, pos: 2875
type: B, layer: 1, pos: 2875
type: A, layer: 1, pos: 2870
type: B, layer: 1, pos: 2870
type: A, layer: 1, pos: 2650
type: B, layer: 1, pos: 2650
type: A, layer: 1, pos: 2868
type: B, layer: 1, pos: 2868
type: A, layer: 1, pos: 2158
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 3527
type: A, layer: 1, pos: 2867
type: B, layer: 1, pos: 2867
type: A, layer: 1, pos: 2188
type: B, layer: 1, pos: 2188
type: A, layer: 1, pos: 2874
type: B, layer: 1, pos: 2874
type: A, layer: 1, pos: 2577
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 3289
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 3100
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 2532
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2228
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2513
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 2203
type: B, layer: 1, pos: 2203
type: A, layer: 1, pos: 2517
type: B, layer: 1, pos: 2517
type: A, layer: 1, pos: 2230
type: B, layer: 1, pos: 2230
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2866
type: B, layer: 1, pos: 2866
type: A, layer: 1, pos: 100
type: B, layer: 1, pos: 100
type: B, layer: 1, pos: 2303
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 2422
type: B, layer: 1, pos: 2499
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2484
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 2422
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 2301
type: B, layer: 1, pos: 2301
type: A, layer: 1, pos: 2302
type: B, layer: 1, pos: 2302
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2662
type: B, layer: 1, pos: 2662
type: A, layer: 1, pos: 2604
type: B, layer: 1, pos: 2604
type: B, layer: 1, pos: 3480
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 2873
type: B, layer: 1, pos: 2873
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2498
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2975
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 2062
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 3510
type: A, layer: 1, pos: 3510
type: A, layer: 1, pos: 2808
type: B, layer: 1, pos: 2808
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2951
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 3290
type: A, layer: 1, pos: 3290
type: B, layer: 1, pos: 2980
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 2559
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 225
type: B, layer: 1, pos: 225
type: B, layer: 1, pos: 2949
type: A, layer: 1, pos: 2949
type: B, layer: 1, pos: 2483
type: A, layer: 1, pos: 2483
type: B, layer: 1, pos: 2844
type: A, layer: 1, pos: 2844
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2087
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2977
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2270
type: B, layer: 1, pos: 2270
type: A, layer: 1, pos: 2057
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2073
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 2959
type: B, layer: 1, pos: 2959
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 2976
type: B, layer: 1, pos: 2976
type: A, layer: 1, pos: 2945
type: B, layer: 1, pos: 2945
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 3525
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 2938
type: B, layer: 1, pos: 2938
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 2053
type: B, layer: 1, pos: 2053
type: A, layer: 1, pos: 2333
type: B, layer: 1, pos: 2333
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 2944
type: B, layer: 1, pos: 2944
type: B, layer: 1, pos: 2858
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 2648
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 3354
type: B, layer: 1, pos: 3354
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 3008
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2338
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 2106
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 330
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 330
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 2479
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 3013
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 3013
type: A, layer: 1, pos: 2547
type: B, layer: 1, pos: 2547
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 2633
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 3305
type: B, layer: 1, pos: 3305
type: A, layer: 1, pos: 2263
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 2263
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 2232
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
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

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3499

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0163890, upper bound: 0.0163616
time: 120.10 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0163890, upper bound: 0.0165212
time: 122.75 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -3.5753701, -2.7684791, -3.5751832, -2.7687304, -0.4761512, 0.4749541
1: -5.3849559, -4.0870261, -5.3830647, -4.0870214, -0.4234920, 0.4204853
2: -0.5319643, -0.3178057, -0.5319294, -0.3178055, -0.0944748, 0.0927913
3: -1.0308781, -0.6424339, -1.0287213, -0.6424780, -0.1172728, 0.1142184
4: -0.5967023, -0.0631320, -0.5966901, -0.0644339, -0.1442420, 0.1469215
5: -0.6400584, -0.2494723, -0.6392505, -0.2494768, -0.1576277, 0.1560817
6: -1.9696503, -1.2726082, -1.9661047, -1.2726089, -0.1236334, 0.1180727
7: 0.5684702, 0.9958918, 0.5685474, 0.9923275, -0.0576927, 0.0650234
8: -5.5624566, -4.4823909, -5.5624580, -4.4831843, -0.3931565, 0.3944101
9: -4.6647401, -3.6479664, -4.6617966, -3.6479661, -0.4123133, 0.4085798

Time for backsubstitution: 6.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3499
type: B, layer: 1, pos: 3499
type: A, layer: 1, pos: 236
type: B, layer: 1, pos: 236
type: B, layer: 1, pos: 3526
type: A, layer: 1, pos: 3526
type: B, layer: 1, pos: 3543
type: A, layer: 1, pos: 3543
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 3095
type: A, layer: 1, pos: 3095
type: A, layer: 1, pos: 3532
type: B, layer: 1, pos: 3532
type: A, layer: 1, pos: 3549
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 3272
type: A, layer: 1, pos: 3272
type: B, layer: 1, pos: 2650
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2868
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 3287
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 2869
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 2870
type: B, layer: 1, pos: 2870
type: A, layer: 1, pos: 2875
type: B, layer: 1, pos: 2875
type: A, layer: 1, pos: 2867
type: B, layer: 1, pos: 2867
type: A, layer: 1, pos: 2158
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 2874
type: A, layer: 1, pos: 2874
type: A, layer: 1, pos: 2188
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2230
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2203
type: B, layer: 1, pos: 2203
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2577
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 2228
type: A, layer: 1, pos: 2866
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 2662
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 2662
type: A, layer: 1, pos: 2228
type: B, layer: 1, pos: 3289
type: A, layer: 1, pos: 3289
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 2422
type: B, layer: 1, pos: 2532
type: A, layer: 1, pos: 2532
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 2422
type: A, layer: 1, pos: 2513
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2517
type: A, layer: 1, pos: 2517
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 2303
type: B, layer: 1, pos: 2303
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 100
type: B, layer: 1, pos: 100
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 2499
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2484
type: B, layer: 1, pos: 2484
type: A, layer: 1, pos: 2301
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 2302
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2604
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2873
type: B, layer: 1, pos: 2975
type: A, layer: 1, pos: 2975
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3480
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 2498
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2062
type: B, layer: 1, pos: 2062
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 2087
type: A, layer: 1, pos: 2073
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 2808
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 3510
type: A, layer: 1, pos: 3510
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 2808
type: B, layer: 1, pos: 225
type: A, layer: 1, pos: 225
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2559
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 51
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2949
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 2483
type: B, layer: 1, pos: 2483
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2270
type: B, layer: 1, pos: 2270
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2057
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 2844
type: B, layer: 1, pos: 3290
type: B, layer: 1, pos: 2977
type: A, layer: 1, pos: 3290
type: B, layer: 1, pos: 2844
type: A, layer: 1, pos: 2959
type: B, layer: 1, pos: 2959
type: A, layer: 1, pos: 2945
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 2858
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 2213
type: B, layer: 1, pos: 2976
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 330
type: B, layer: 1, pos: 2648
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2938
type: A, layer: 1, pos: 2938
type: B, layer: 1, pos: 3525
type: A, layer: 1, pos: 2944
type: B, layer: 1, pos: 2944
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2333
type: B, layer: 1, pos: 2333
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 2547
type: B, layer: 1, pos: 2547
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 2053
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2232
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2338
type: B, layer: 1, pos: 2338
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 3008
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 3354
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 2106
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 3305
type: B, layer: 1, pos: 3305
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 3013
type: B, layer: 1, pos: 3013
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 2858
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 2263
type: B, layer: 1, pos: 2263
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 88
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
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2648
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 330
type: A, layer: 1, pos: 2232

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 3499

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165143, upper bound: 0.0163630
time: 30.42 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165145, upper bound: 0.0165322
time: 9.75 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 46.49 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 46.49
Output dim: 7, lower bound: -0.0162035, upper bound: 0.0163651
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 46.49
Output dim: 7, lower bound: -0.0162036, upper bound: 0.0165234
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 46.49
Output dim: 7, lower bound: -0.0163287, upper bound: 0.0163654
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 46.49
Output dim: 7, lower bound: -0.0163286, upper bound: 0.0165245
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 46.49
Output dim: 7, lower bound: -0.0163890, upper bound: 0.0163616
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 46.49
Output dim: 7, lower bound: -0.0163890, upper bound: 0.0165212
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 46.49
Output dim: 7, lower bound: -0.0165143, upper bound: 0.0163630
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 46.49
Output dim: 7, lower bound: -0.0165145, upper bound: 0.0165322

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3.5735550, -2.7687328, -3.5730605, -2.7696753, -0.4717057, 0.4728150
1: -5.3830643, -4.0880795, -5.3830585, -4.0879383, -0.4202361, 0.4203864
2: -0.5311763, -0.3178269, -0.5310703, -0.3185923, -0.0927804, 0.0933223
3: -1.0287111, -0.6442580, -1.0282254, -0.6444534, -0.1136381, 0.1135575
4: -0.5953497, -0.0644430, -0.5952510, -0.0649008, -0.1442943, 0.1441723
5: -0.6391110, -0.2500225, -0.6390710, -0.2501235, -0.1556101, 0.1556372
6: -1.9656240, -1.2746180, -1.9655517, -1.2747809, -0.1179247, 0.1178975
7: 0.5705575, 0.9909223, 0.5723066, 0.9907237, -0.0587492, 0.0530753
8: -5.5619812, -4.4836025, -5.5616751, -4.4837208, -0.3913912, 0.3925529
9: -4.6614690, -3.6502423, -4.6610084, -3.6501050, -0.4076304, 0.4070210

Time for backsubstitution: 6.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 236
type: A, layer: 1, pos: 236
type: B, layer: 1, pos: 3526
type: A, layer: 1, pos: 3526
type: B, layer: 1, pos: 3543
type: A, layer: 1, pos: 3543
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 3095
type: B, layer: 1, pos: 3095
type: A, layer: 1, pos: 3532
type: B, layer: 1, pos: 3532
type: B, layer: 1, pos: 3272
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 3549
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 2869
type: B, layer: 1, pos: 2869
type: A, layer: 1, pos: 2875
type: B, layer: 1, pos: 2875
type: A, layer: 1, pos: 2870
type: B, layer: 1, pos: 2870
type: A, layer: 1, pos: 2650
type: B, layer: 1, pos: 2650
type: A, layer: 1, pos: 2868
type: B, layer: 1, pos: 2868
type: A, layer: 1, pos: 2158
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 3527
type: A, layer: 1, pos: 2867
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 3499
type: A, layer: 1, pos: 2188
type: B, layer: 1, pos: 2188
type: A, layer: 1, pos: 2874
type: B, layer: 1, pos: 2874
type: A, layer: 1, pos: 2577
type: B, layer: 1, pos: 2577
type: A, layer: 1, pos: 3289
type: B, layer: 1, pos: 3289
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 3100
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 2532
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2228
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2513
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 2203
type: B, layer: 1, pos: 2203
type: A, layer: 1, pos: 2517
type: B, layer: 1, pos: 2517
type: A, layer: 1, pos: 2230
type: B, layer: 1, pos: 2230
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2866
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 100
type: A, layer: 1, pos: 100
type: A, layer: 1, pos: 2303
type: B, layer: 1, pos: 2303
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 2422
type: B, layer: 1, pos: 2499
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2484
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 2422
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 2301
type: B, layer: 1, pos: 2301
type: A, layer: 1, pos: 2302
type: B, layer: 1, pos: 2302
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2662
type: B, layer: 1, pos: 2662
type: B, layer: 1, pos: 2604
type: A, layer: 1, pos: 2604
type: B, layer: 1, pos: 3480
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 2873
type: B, layer: 1, pos: 2873
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2498
type: B, layer: 1, pos: 2498
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2975
type: B, layer: 1, pos: 2975
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3510
type: A, layer: 1, pos: 3510
type: A, layer: 1, pos: 2062
type: B, layer: 1, pos: 2062
type: A, layer: 1, pos: 2808
type: B, layer: 1, pos: 2808
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 3290
type: A, layer: 1, pos: 3290
type: B, layer: 1, pos: 2980
type: A, layer: 1, pos: 2980
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 225
type: B, layer: 1, pos: 225
type: B, layer: 1, pos: 2949
type: A, layer: 1, pos: 2949
type: B, layer: 1, pos: 2483
type: A, layer: 1, pos: 2483
type: B, layer: 1, pos: 2844
type: A, layer: 1, pos: 2844
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2087
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2977
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2270
type: B, layer: 1, pos: 2270
type: A, layer: 1, pos: 2057
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2073
type: B, layer: 1, pos: 2073
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 2959
type: B, layer: 1, pos: 2959
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 2976
type: B, layer: 1, pos: 2976
type: A, layer: 1, pos: 2945
type: B, layer: 1, pos: 2945
type: A, layer: 1, pos: 3525
type: B, layer: 1, pos: 3525
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 2938
type: B, layer: 1, pos: 2938
type: B, layer: 1, pos: 2138
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2053
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2333
type: A, layer: 1, pos: 2333
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2944
type: B, layer: 1, pos: 2944
type: A, layer: 1, pos: 2858
type: B, layer: 1, pos: 2858
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 3354
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3008
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2338
type: B, layer: 1, pos: 2338
type: A, layer: 1, pos: 330
type: A, layer: 1, pos: 2106
type: B, layer: 1, pos: 2106
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 330
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 2479
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 3013
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 3013
type: A, layer: 1, pos: 2547
type: B, layer: 1, pos: 2547
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 2633
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 3305
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 3305
type: A, layer: 1, pos: 2263
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 2263
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2232
type: B, layer: 1, pos: 2232
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
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
type: B, layer: 1, pos: 236

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0159832, upper bound: 0.0165240
time: 28.98 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0162029, upper bound: 0.0165175
time: 74.67 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.5748627, -2.7687314, -3.5749109, -2.7687316, -0.4742160, 0.4741971
1: -5.3830600, -4.0880761, -5.3830590, -4.0879426, -0.4202533, 0.4203918
2: -0.5319063, -0.3178264, -0.5319092, -0.3178236, -0.0941394, 0.0927021
3: -1.0287166, -0.6437092, -1.0287172, -0.6435574, -0.1139767, 0.1140029
4: -0.5959550, -0.0644432, -0.5960549, -0.0644414, -0.1440920, 0.1449894
5: -0.6392361, -0.2499312, -0.6392368, -0.2498777, -0.1563647, 0.1556711
6: -1.9661026, -1.2746170, -1.9661033, -1.2743422, -0.1182973, 0.1178885
7: 0.5705535, 0.9923134, 0.5702824, 0.9923145, -0.0573930, 0.0565697
8: -5.5619855, -4.4832726, -5.5620513, -4.4832630, -0.3915249, 0.3931929
9: -4.6616168, -3.6496758, -4.6616416, -3.6494379, -0.4078459, 0.4079274

Time for backsubstitution: 5.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 236
type: B, layer: 1, pos: 236
type: A, layer: 1, pos: 3526
type: B, layer: 1, pos: 3526
type: A, layer: 1, pos: 3543
type: B, layer: 1, pos: 3543
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 3095
type: A, layer: 1, pos: 3095
type: A, layer: 1, pos: 3532
type: B, layer: 1, pos: 3532
type: A, layer: 1, pos: 3549
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 3272
type: A, layer: 1, pos: 3272
type: B, layer: 1, pos: 2650
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 2868
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 3287
type: B, layer: 1, pos: 2869
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 2870
type: B, layer: 1, pos: 2870
type: A, layer: 1, pos: 2875
type: B, layer: 1, pos: 2875
type: A, layer: 1, pos: 2867
type: B, layer: 1, pos: 2867
type: A, layer: 1, pos: 2158
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 2874
type: A, layer: 1, pos: 2874
type: A, layer: 1, pos: 2188
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 3499
type: B, layer: 1, pos: 2230
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2203
type: B, layer: 1, pos: 2203
type: A, layer: 1, pos: 2577
type: B, layer: 1, pos: 2577
type: A, layer: 1, pos: 3100
type: B, layer: 1, pos: 2866
type: A, layer: 1, pos: 2866
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 2662
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 2662
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 3289
type: A, layer: 1, pos: 3289
type: B, layer: 1, pos: 2532
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2422
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2422
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 2633
type: A, layer: 1, pos: 2513
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2517
type: A, layer: 1, pos: 2517
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 2303
type: B, layer: 1, pos: 2303
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 100
type: A, layer: 1, pos: 100
type: B, layer: 1, pos: 2499
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2484
type: B, layer: 1, pos: 2484
type: A, layer: 1, pos: 2301
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 2302
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2604
type: A, layer: 1, pos: 2604
type: B, layer: 1, pos: 2975
type: A, layer: 1, pos: 2975
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3480
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 2498
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2062
type: B, layer: 1, pos: 2062
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2087
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2087
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2951
type: A, layer: 1, pos: 2073
type: A, layer: 1, pos: 3510
type: B, layer: 1, pos: 3510
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2980
type: A, layer: 1, pos: 2808
type: A, layer: 1, pos: 2980
type: B, layer: 1, pos: 2808
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 225
type: A, layer: 1, pos: 225
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 51
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2949
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 2483
type: B, layer: 1, pos: 2483
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2057
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2270
type: B, layer: 1, pos: 2270
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2057
type: A, layer: 1, pos: 2977
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 3290
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2844
type: B, layer: 1, pos: 2844
type: A, layer: 1, pos: 2959
type: B, layer: 1, pos: 2959
type: A, layer: 1, pos: 2945
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 2976
type: A, layer: 1, pos: 2976
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 3525
type: B, layer: 1, pos: 3525
type: B, layer: 1, pos: 2938
type: A, layer: 1, pos: 2938
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2944
type: B, layer: 1, pos: 2944
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2858
type: A, layer: 1, pos: 330
type: B, layer: 1, pos: 2333
type: A, layer: 1, pos: 2333
type: A, layer: 1, pos: 2213
type: B, layer: 1, pos: 2138
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 2547
type: B, layer: 1, pos: 2547
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: B, layer: 1, pos: 2648
type: A, layer: 1, pos: 2053
type: B, layer: 1, pos: 2053
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2338
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 3354
type: B, layer: 1, pos: 3354
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 2858
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2106
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 2479
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2479
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 3305
type: B, layer: 1, pos: 3305
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: B, layer: 1, pos: 2232
type: A, layer: 1, pos: 3013
type: B, layer: 1, pos: 3013
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: A, layer: 1, pos: 2648
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 2263
type: B, layer: 1, pos: 2263
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 88
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
type: B, layer: 1, pos: 330
type: A, layer: 1, pos: 2232

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 236

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0163282, upper bound: 0.0163022
time: 117.21 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0163278, upper bound: 0.0165198
time: 16.73 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.5740597, -2.7684798, -3.5733283, -2.7696743, -0.4731921, 0.4735450
1: -5.3849587, -4.0870299, -5.3830638, -4.0870171, -0.4231690, 0.4204738
2: -0.5312344, -0.3178103, -0.5310907, -0.3185774, -0.0931158, 0.0933627
3: -1.0308719, -0.6429965, -1.0282300, -0.6433868, -0.1169257, 0.1137442
4: -0.5960968, -0.0631342, -0.5958858, -0.0648948, -0.1444237, 0.1461184
5: -0.6399332, -0.2495675, -0.6390851, -0.2497262, -0.1568582, 0.1557503
6: -1.9691705, -1.2726095, -1.9655523, -1.2730476, -0.1232032, 0.1180814
7: 0.5684741, 0.9944922, 0.5705718, 0.9907295, -0.0590423, 0.0584014
8: -5.5624442, -4.4827213, -5.5620766, -4.4836426, -0.3916401, 0.3937435
9: -4.6645932, -3.6485579, -4.6611652, -3.6486559, -0.4121560, 0.4076524

Time for backsubstitution: 5.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 236
type: B, layer: 1, pos: 236
type: B, layer: 1, pos: 3526
type: A, layer: 1, pos: 3526
type: B, layer: 1, pos: 3543
type: A, layer: 1, pos: 3543
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 3095
type: B, layer: 1, pos: 3095
type: A, layer: 1, pos: 3532
type: B, layer: 1, pos: 3532
type: B, layer: 1, pos: 3272
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 3549
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 3287
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 2869
type: B, layer: 1, pos: 2869
type: A, layer: 1, pos: 2875
type: B, layer: 1, pos: 2875
type: A, layer: 1, pos: 2870
type: B, layer: 1, pos: 2870
type: A, layer: 1, pos: 2650
type: B, layer: 1, pos: 2650
type: A, layer: 1, pos: 2868
type: B, layer: 1, pos: 2868
type: A, layer: 1, pos: 2158
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 3527
type: A, layer: 1, pos: 2867
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 3499
type: A, layer: 1, pos: 2188
type: B, layer: 1, pos: 2188
type: A, layer: 1, pos: 2874
type: B, layer: 1, pos: 2874
type: A, layer: 1, pos: 2577
type: B, layer: 1, pos: 2577
type: A, layer: 1, pos: 3289
type: B, layer: 1, pos: 3289
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 3100
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 2532
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2228
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2513
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 2203
type: B, layer: 1, pos: 2203
type: A, layer: 1, pos: 2517
type: B, layer: 1, pos: 2517
type: A, layer: 1, pos: 2230
type: B, layer: 1, pos: 2230
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2866
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 100
type: A, layer: 1, pos: 100
type: B, layer: 1, pos: 2303
type: A, layer: 1, pos: 2303
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 2422
type: B, layer: 1, pos: 2499
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2484
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 2422
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 2301
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2302
type: B, layer: 1, pos: 2302
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2662
type: B, layer: 1, pos: 2662
type: B, layer: 1, pos: 2604
type: A, layer: 1, pos: 2604
type: B, layer: 1, pos: 3480
type: A, layer: 1, pos: 3480
type: B, layer: 1, pos: 2873
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2498
type: B, layer: 1, pos: 2498
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2975
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2062
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 3510
type: A, layer: 1, pos: 3510
type: B, layer: 1, pos: 2808
type: A, layer: 1, pos: 2808
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2951
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 3290
type: A, layer: 1, pos: 3290
type: B, layer: 1, pos: 2980
type: A, layer: 1, pos: 2980
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 225
type: B, layer: 1, pos: 225
type: B, layer: 1, pos: 2949
type: A, layer: 1, pos: 2949
type: B, layer: 1, pos: 2483
type: A, layer: 1, pos: 2483
type: B, layer: 1, pos: 2844
type: A, layer: 1, pos: 2844
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2087
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2977
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2270
type: B, layer: 1, pos: 2270
type: A, layer: 1, pos: 2057
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2073
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 2959
type: B, layer: 1, pos: 2959
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 2976
type: B, layer: 1, pos: 2976
type: A, layer: 1, pos: 2945
type: B, layer: 1, pos: 2945
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 3525
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 2938
type: B, layer: 1, pos: 2938
type: B, layer: 1, pos: 2138
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2053
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2333
type: A, layer: 1, pos: 2333
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2944
type: B, layer: 1, pos: 2944
type: B, layer: 1, pos: 2858
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 330
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 3354
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3008
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2338
type: B, layer: 1, pos: 2338
type: A, layer: 1, pos: 2106
type: B, layer: 1, pos: 2106
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 2479
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 3013
type: B, layer: 1, pos: 330
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 3013
type: A, layer: 1, pos: 2547
type: B, layer: 1, pos: 2547
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 2633
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 3305
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 3305
type: A, layer: 1, pos: 2263
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 2263
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 2232
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
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
type: A, layer: 1, pos: 236

## Relational analysis of NS_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0163884, upper bound: 0.0163067
time: 6.30 seconds

## Relational analysis of NS_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0163886, upper bound: 0.0165271
time: 32.44 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.5747743, -2.7687249, -3.5746913, -2.7687321, -0.4757031, 0.4745134
1: -5.3846679, -4.0871482, -5.3828244, -4.0870218, -0.4232366, 0.4201498
2: -0.5319493, -0.3178571, -0.5319238, -0.3178505, -0.0943866, 0.0927230
3: -1.0309161, -0.6425670, -1.0287194, -0.6425112, -0.1172286, 0.1140416
4: -0.5966629, -0.0631670, -0.5966847, -0.0644635, -0.1441880, 0.1468884
5: -0.6399103, -0.2498233, -0.6392434, -0.2497737, -0.1572696, 0.1557884
6: -1.9695998, -1.2726235, -1.9660625, -1.2726090, -0.1235821, 0.1180114
7: 0.5699326, 0.9929307, 0.5685526, 0.9897882, -0.0536519, 0.0620396
8: -5.5612426, -4.4829445, -5.5614138, -4.4832129, -0.3920457, 0.3929386
9: -4.6644711, -3.6483850, -4.6615944, -3.6483383, -0.4114512, 0.4079511

Time for backsubstitution: 6.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 236
type: B, layer: 1, pos: 236
type: B, layer: 1, pos: 3526
type: A, layer: 1, pos: 3526
type: B, layer: 1, pos: 3543
type: A, layer: 1, pos: 3543
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 3095
type: A, layer: 1, pos: 3095
type: A, layer: 1, pos: 3532
type: B, layer: 1, pos: 3532
type: A, layer: 1, pos: 3549
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 3272
type: A, layer: 1, pos: 3272
type: B, layer: 1, pos: 2650
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2868
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 3287
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 2869
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 2870
type: B, layer: 1, pos: 2870
type: A, layer: 1, pos: 2875
type: B, layer: 1, pos: 2875
type: A, layer: 1, pos: 2867
type: B, layer: 1, pos: 2867
type: A, layer: 1, pos: 2158
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 2874
type: A, layer: 1, pos: 2874
type: A, layer: 1, pos: 2188
type: B, layer: 1, pos: 3499
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2230
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 2203
type: B, layer: 1, pos: 2203
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 2577
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 2866
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 2662
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 2662
type: A, layer: 1, pos: 2228
type: B, layer: 1, pos: 3289
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 2422
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 2532
type: A, layer: 1, pos: 2532
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 2422
type: A, layer: 1, pos: 2513
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2517
type: A, layer: 1, pos: 2517
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 2303
type: B, layer: 1, pos: 2303
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 100
type: B, layer: 1, pos: 100
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 2499
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2484
type: B, layer: 1, pos: 2484
type: A, layer: 1, pos: 2301
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2873
type: B, layer: 1, pos: 2302
type: A, layer: 1, pos: 2302
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2604
type: A, layer: 1, pos: 2604
type: A, layer: 1, pos: 2873
type: B, layer: 1, pos: 2975
type: A, layer: 1, pos: 2975
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3480
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 2498
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2062
type: B, layer: 1, pos: 2062
type: A, layer: 1, pos: 2087
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 2087
type: A, layer: 1, pos: 2073
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 2808
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 3510
type: A, layer: 1, pos: 3510
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 2073
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 776
type: B, layer: 1, pos: 2808
type: B, layer: 1, pos: 225
type: A, layer: 1, pos: 225
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2559
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 51
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2949
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 2483
type: B, layer: 1, pos: 2483
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2270
type: B, layer: 1, pos: 2270
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2057
type: A, layer: 1, pos: 2844
type: A, layer: 1, pos: 2977
type: B, layer: 1, pos: 3290
type: A, layer: 1, pos: 3290
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 2844
type: A, layer: 1, pos: 2959
type: A, layer: 1, pos: 2213
type: B, layer: 1, pos: 2858
type: A, layer: 1, pos: 2945
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 2959
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2976
type: A, layer: 1, pos: 330
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2105
type: B, layer: 1, pos: 2938
type: A, layer: 1, pos: 2938
type: B, layer: 1, pos: 3525
type: A, layer: 1, pos: 2944
type: B, layer: 1, pos: 2944
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2333
type: B, layer: 1, pos: 2333
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2138
type: B, layer: 1, pos: 2232
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 2547
type: B, layer: 1, pos: 2547
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 2053
type: B, layer: 1, pos: 2053
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2338
type: B, layer: 1, pos: 2338
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 3354
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 2106
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 3305
type: B, layer: 1, pos: 3305
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 3013
type: B, layer: 1, pos: 3013
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 2263
type: B, layer: 1, pos: 2263
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
type: B, layer: 1, pos: 88
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
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 2648
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 330
type: A, layer: 1, pos: 2232

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 236

## Relational analysis of NS_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165139, upper bound: 0.0161483
time: 39.86 seconds

## Relational analysis of NS_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165138, upper bound: 0.0163669
time: 10.99 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.5753653, -2.7684791, -3.5751793, -2.7687304, -0.4757025, 0.4749277
1: -5.3849549, -4.0870261, -5.3830633, -4.0870214, -0.4231862, 0.4204793
2: -0.5319643, -0.3178099, -0.5319295, -0.3178089, -0.0944748, 0.0927425
3: -1.0308781, -0.6424484, -1.0287215, -0.6424901, -0.1172642, 0.1141882
4: -0.5967025, -0.0631337, -0.5966902, -0.0644356, -0.1442216, 0.1469355
5: -0.6400583, -0.2494764, -0.6392503, -0.2494807, -0.1576127, 0.1557843
6: -1.9696488, -1.2726082, -1.9661033, -1.2726089, -0.1235761, 0.1180722
7: 0.5684702, 0.9958838, 0.5685474, 0.9923204, -0.0576859, 0.0609162
8: -5.5624480, -4.4823904, -5.5624514, -4.4831839, -0.3917738, 0.3943835
9: -4.6647396, -3.6479921, -4.6617966, -3.6479886, -0.4123719, 0.4085581

Time for backsubstitution: 6.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 236
type: B, layer: 1, pos: 236
type: B, layer: 1, pos: 3526
type: A, layer: 1, pos: 3526
type: B, layer: 1, pos: 3543
type: A, layer: 1, pos: 3543
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 3095
type: A, layer: 1, pos: 3095
type: A, layer: 1, pos: 3532
type: B, layer: 1, pos: 3532
type: A, layer: 1, pos: 3549
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 3272
type: A, layer: 1, pos: 3272
type: B, layer: 1, pos: 2650
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 2868
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 3287
type: B, layer: 1, pos: 601
type: B, layer: 1, pos: 2869
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 2870
type: B, layer: 1, pos: 2870
type: A, layer: 1, pos: 2875
type: B, layer: 1, pos: 2875
type: A, layer: 1, pos: 2867
type: B, layer: 1, pos: 2867
type: A, layer: 1, pos: 2158
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 2874
type: A, layer: 1, pos: 2874
type: B, layer: 1, pos: 3499
type: A, layer: 1, pos: 2188
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2230
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2203
type: B, layer: 1, pos: 2203
type: A, layer: 1, pos: 2577
type: B, layer: 1, pos: 2577
type: A, layer: 1, pos: 3100
type: B, layer: 1, pos: 2866
type: A, layer: 1, pos: 2866
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2662
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 2662
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 3289
type: A, layer: 1, pos: 3289
type: B, layer: 1, pos: 2532
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2422
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2422
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 2633
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 2633
type: A, layer: 1, pos: 2513
type: B, layer: 1, pos: 2513
type: B, layer: 1, pos: 2517
type: A, layer: 1, pos: 2517
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 2303
type: B, layer: 1, pos: 2303
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 100
type: A, layer: 1, pos: 100
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 2499
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2484
type: B, layer: 1, pos: 2484
type: A, layer: 1, pos: 2301
type: B, layer: 1, pos: 2301
type: B, layer: 1, pos: 2302
type: A, layer: 1, pos: 2302
type: B, layer: 1, pos: 2873
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2604
type: A, layer: 1, pos: 2604
type: B, layer: 1, pos: 2975
type: A, layer: 1, pos: 2975
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3480
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 2498
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2062
type: B, layer: 1, pos: 2062
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 2087
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2087
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2951
type: A, layer: 1, pos: 2073
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2980
type: B, layer: 1, pos: 3510
type: A, layer: 1, pos: 2808
type: A, layer: 1, pos: 3510
type: B, layer: 1, pos: 2808
type: A, layer: 1, pos: 2980
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 2072
type: B, layer: 1, pos: 225
type: A, layer: 1, pos: 225
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 51
type: A, layer: 1, pos: 51
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2838
type: B, layer: 1, pos: 2949
type: A, layer: 1, pos: 2949
type: B, layer: 1, pos: 2483
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 2270
type: B, layer: 1, pos: 2270
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2057
type: A, layer: 1, pos: 2977
type: B, layer: 1, pos: 2977
type: B, layer: 1, pos: 3290
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2844
type: B, layer: 1, pos: 2844
type: A, layer: 1, pos: 2959
type: B, layer: 1, pos: 2959
type: A, layer: 1, pos: 2945
type: B, layer: 1, pos: 2945
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 2976
type: A, layer: 1, pos: 2976
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: B, layer: 1, pos: 2938
type: A, layer: 1, pos: 2938
type: B, layer: 1, pos: 3525
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2944
type: B, layer: 1, pos: 2944
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 330
type: B, layer: 1, pos: 2858
type: B, layer: 1, pos: 2333
type: A, layer: 1, pos: 2333
type: B, layer: 1, pos: 2138
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2547
type: B, layer: 1, pos: 2547
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 2053
type: B, layer: 1, pos: 2053
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2338
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: B, layer: 1, pos: 3008
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 3354
type: B, layer: 1, pos: 3354
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2106
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 2479
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 3305
type: B, layer: 1, pos: 3305
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 3013
type: B, layer: 1, pos: 2213
type: A, layer: 1, pos: 2648
type: B, layer: 1, pos: 3013
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 2232
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 2263
type: B, layer: 1, pos: 2263
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 88
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
type: B, layer: 1, pos: 330
type: A, layer: 1, pos: 2232

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 236

## Relational analysis of NS_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165136, upper bound: 0.0163029
time: 4.31 seconds

## Relational analysis of NS_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0165134, upper bound: 0.0165157
time: 72.86 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 83.33 seconds
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 83.33
Output dim: 7, lower bound: -0.0159832, upper bound: 0.0165240
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 83.33
Output dim: 7, lower bound: -0.0162029, upper bound: 0.0165175
NS_A1_B2_A2_A1, status: Status.VERIFIED, split count: 4, time: 83.33
Output dim: 7, lower bound: -0.0163282, upper bound: 0.0163022
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 83.33
Output dim: 7, lower bound: -0.0163278, upper bound: 0.0165198
NS_A2_B1_A2_A1, status: Status.VERIFIED, split count: 4, time: 83.33
Output dim: 7, lower bound: -0.0163884, upper bound: 0.0163067
NS_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 83.33
Output dim: 7, lower bound: -0.0163886, upper bound: 0.0165271
NS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 83.33
Output dim: 7, lower bound: -0.0165139, upper bound: 0.0161483
NS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 83.33
Output dim: 7, lower bound: -0.0165138, upper bound: 0.0163669
NS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 83.33
Output dim: 7, lower bound: -0.0165136, upper bound: 0.0163029
NS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 83.33
Output dim: 7, lower bound: -0.0165134, upper bound: 0.0165157

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3.5728555, -2.7687333, -3.5721486, -2.7698088, -0.4708351, 0.4718746
1: -5.3830590, -4.0914950, -5.3822575, -4.0922432, -0.4158953, 0.4161121
2: -0.5237662, -0.3178270, -0.5217422, -0.3203329, -0.0837573, 0.0840624
3: -1.0284770, -0.6442988, -1.0279231, -0.6445675, -0.1133148, 0.1132388
4: -0.5913297, -0.0644430, -0.5901853, -0.0658282, -0.1392215, 0.1391680
5: -0.6360487, -0.2500629, -0.6352023, -0.2508768, -0.1517337, 0.1516964
6: -1.9606303, -1.2746185, -1.9592649, -1.2759050, -0.1117088, 0.1115945
7: 0.5705599, 0.9867558, 0.5732892, 0.9854772, -0.0534871, 0.0478925
8: -5.5619798, -4.4913468, -5.5598650, -4.4935513, -0.3814578, 0.3828136
9: -4.6613331, -3.6635695, -4.6578374, -3.6668046, -0.3913082, 0.3910878

Time for backsubstitution: 6.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3526
type: A, layer: 1, pos: 3526
type: B, layer: 1, pos: 3543
type: A, layer: 1, pos: 3543
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 3095
type: B, layer: 1, pos: 3095
type: A, layer: 1, pos: 3532
type: B, layer: 1, pos: 3532
type: B, layer: 1, pos: 3272
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 3549
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 2869
type: B, layer: 1, pos: 2869
type: A, layer: 1, pos: 2875
type: B, layer: 1, pos: 2875
type: A, layer: 1, pos: 2870
type: B, layer: 1, pos: 2870
type: A, layer: 1, pos: 2650
type: B, layer: 1, pos: 2650
type: A, layer: 1, pos: 2868
type: B, layer: 1, pos: 2868
type: A, layer: 1, pos: 2158
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 3527
type: A, layer: 1, pos: 236
type: A, layer: 1, pos: 2867
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 3499
type: A, layer: 1, pos: 2188
type: B, layer: 1, pos: 2188
type: A, layer: 1, pos: 2874
type: B, layer: 1, pos: 2874
type: A, layer: 1, pos: 2577
type: B, layer: 1, pos: 2577
type: A, layer: 1, pos: 3289
type: B, layer: 1, pos: 3289
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 3100
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 2532
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2228
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2513
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 2203
type: B, layer: 1, pos: 2203
type: A, layer: 1, pos: 2517
type: B, layer: 1, pos: 2517
type: A, layer: 1, pos: 2230
type: B, layer: 1, pos: 2230
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2866
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 100
type: A, layer: 1, pos: 100
type: A, layer: 1, pos: 2303
type: B, layer: 1, pos: 2303
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 2422
type: B, layer: 1, pos: 2499
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2484
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 2422
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 2301
type: B, layer: 1, pos: 2301
type: A, layer: 1, pos: 2302
type: B, layer: 1, pos: 2302
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2662
type: B, layer: 1, pos: 2662
type: B, layer: 1, pos: 2604
type: A, layer: 1, pos: 2604
type: B, layer: 1, pos: 3480
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 2873
type: B, layer: 1, pos: 2873
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2498
type: B, layer: 1, pos: 2498
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2975
type: B, layer: 1, pos: 2975
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3510
type: A, layer: 1, pos: 3510
type: A, layer: 1, pos: 2062
type: B, layer: 1, pos: 2062
type: A, layer: 1, pos: 2808
type: B, layer: 1, pos: 2808
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 3290
type: A, layer: 1, pos: 3290
type: B, layer: 1, pos: 2980
type: A, layer: 1, pos: 2980
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 225
type: B, layer: 1, pos: 225
type: B, layer: 1, pos: 2949
type: A, layer: 1, pos: 2949
type: B, layer: 1, pos: 2483
type: A, layer: 1, pos: 2483
type: B, layer: 1, pos: 2844
type: A, layer: 1, pos: 2844
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2087
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2977
type: A, layer: 1, pos: 2977
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2270
type: B, layer: 1, pos: 2270
type: A, layer: 1, pos: 2057
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2073
type: B, layer: 1, pos: 2073
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 2959
type: B, layer: 1, pos: 2959
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 2976
type: B, layer: 1, pos: 2976
type: A, layer: 1, pos: 2945
type: B, layer: 1, pos: 2945
type: A, layer: 1, pos: 3525
type: B, layer: 1, pos: 3525
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 2938
type: B, layer: 1, pos: 2938
type: B, layer: 1, pos: 2138
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2053
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2333
type: A, layer: 1, pos: 2333
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2944
type: B, layer: 1, pos: 2944
type: A, layer: 1, pos: 2858
type: B, layer: 1, pos: 2858
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 3354
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 3354
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 3008
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2338
type: B, layer: 1, pos: 2338
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 330
type: B, layer: 1, pos: 2106
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: B, layer: 1, pos: 330
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 2479
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 2337
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 3013
type: B, layer: 1, pos: 2213
type: B, layer: 1, pos: 3013
type: A, layer: 1, pos: 2547
type: B, layer: 1, pos: 2547
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 2633
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 87
type: A, layer: 1, pos: 3305
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 3305
type: A, layer: 1, pos: 2263
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 2263
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2232
type: B, layer: 1, pos: 2232
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
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

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3526

## Relational analysis of NS_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0159821, upper bound: 0.0164149
time: 86.37 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0159825, upper bound: 0.0165309
time: 4.52 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3.5735521, -2.7687328, -3.5730577, -2.7696750, -0.4717047, 0.4724348
1: -5.3830643, -4.0880842, -5.3830600, -4.0879436, -0.4153044, 0.4203860
2: -0.5311705, -0.3178269, -0.5310627, -0.3185922, -0.0927737, 0.0828086
3: -1.0287105, -0.6442580, -1.0282249, -0.6444538, -0.1136377, 0.1135211
4: -0.5953466, -0.0644429, -0.5952467, -0.0649008, -0.1442524, 0.1387000
5: -0.6391048, -0.2500226, -0.6390631, -0.2501236, -0.1556085, 0.1519658
6: -1.9656138, -1.2746180, -1.9655380, -1.2747808, -0.1179165, 0.1107072
7: 0.5705575, 0.9909191, 0.5723066, 0.9907193, -0.0527371, 0.0530753
8: -5.5619812, -4.4836140, -5.5616751, -4.4837356, -0.3815604, 0.3925526
9: -4.6614614, -3.6502516, -4.6609983, -3.6501184, -0.3894581, 0.4070121

Time for backsubstitution: 6.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3526
type: A, layer: 1, pos: 3526
type: B, layer: 1, pos: 3543
type: A, layer: 1, pos: 3543
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: A, layer: 1, pos: 3095
type: B, layer: 1, pos: 3095
type: A, layer: 1, pos: 3532
type: B, layer: 1, pos: 3532
type: B, layer: 1, pos: 3272
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 3549
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 3287
type: A, layer: 1, pos: 2869
type: B, layer: 1, pos: 2869
type: A, layer: 1, pos: 2875
type: B, layer: 1, pos: 2875
type: A, layer: 1, pos: 2870
type: B, layer: 1, pos: 2870
type: A, layer: 1, pos: 2650
type: B, layer: 1, pos: 2650
type: A, layer: 1, pos: 236
type: A, layer: 1, pos: 2868
type: B, layer: 1, pos: 2868
type: A, layer: 1, pos: 2158
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 3527
type: A, layer: 1, pos: 2867
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 3499
type: A, layer: 1, pos: 2188
type: B, layer: 1, pos: 2188
type: A, layer: 1, pos: 2874
type: B, layer: 1, pos: 2874
type: B, layer: 1, pos: 2577
type: A, layer: 1, pos: 2577
type: B, layer: 1, pos: 3289
type: A, layer: 1, pos: 3289
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 3387
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 2532
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2228
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2513
type: B, layer: 1, pos: 2513
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 2203
type: B, layer: 1, pos: 2203
type: A, layer: 1, pos: 2517
type: B, layer: 1, pos: 2517
type: A, layer: 1, pos: 2230
type: B, layer: 1, pos: 2230
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 2866
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 100
type: A, layer: 1, pos: 100
type: A, layer: 1, pos: 2303
type: B, layer: 1, pos: 2303
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 2422
type: A, layer: 1, pos: 2499
type: B, layer: 1, pos: 2499
type: B, layer: 1, pos: 2484
type: A, layer: 1, pos: 2484
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 2301
type: B, layer: 1, pos: 2301
type: A, layer: 1, pos: 2302
type: B, layer: 1, pos: 2302
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 99
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 2662
type: A, layer: 1, pos: 2662
type: B, layer: 1, pos: 2604
type: A, layer: 1, pos: 2604
type: B, layer: 1, pos: 3480
type: A, layer: 1, pos: 3480
type: A, layer: 1, pos: 2873
type: B, layer: 1, pos: 2873
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 2498
type: B, layer: 1, pos: 2498
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2975
type: B, layer: 1, pos: 2975
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 3024
type: B, layer: 1, pos: 3510
type: A, layer: 1, pos: 3510
type: A, layer: 1, pos: 2062
type: B, layer: 1, pos: 2062
type: B, layer: 1, pos: 2808
type: A, layer: 1, pos: 2808
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 3290
type: A, layer: 1, pos: 3290
type: B, layer: 1, pos: 2980
type: A, layer: 1, pos: 2980
type: A, layer: 1, pos: 2559
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 225
type: B, layer: 1, pos: 225
type: B, layer: 1, pos: 2949
type: A, layer: 1, pos: 2949
type: B, layer: 1, pos: 2483
type: A, layer: 1, pos: 2483
type: B, layer: 1, pos: 2844
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2844
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2514
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2087
type: B, layer: 1, pos: 2087
type: A, layer: 1, pos: 2977
type: B, layer: 1, pos: 2977
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2270
type: B, layer: 1, pos: 2270
type: A, layer: 1, pos: 2057
type: B, layer: 1, pos: 2057
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2073
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 3241
type: A, layer: 1, pos: 3241
type: A, layer: 1, pos: 2959
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: B, layer: 1, pos: 2959
type: B, layer: 1, pos: 2976
type: A, layer: 1, pos: 2976
type: A, layer: 1, pos: 330
type: A, layer: 1, pos: 2945
type: B, layer: 1, pos: 2945
type: A, layer: 1, pos: 3525
type: B, layer: 1, pos: 3525
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 2938
type: A, layer: 1, pos: 2938
type: B, layer: 1, pos: 2138
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 2053
type: B, layer: 1, pos: 2053
type: A, layer: 1, pos: 2333
type: B, layer: 1, pos: 2333
type: A, layer: 1, pos: 83
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 2858
type: A, layer: 1, pos: 2944
type: B, layer: 1, pos: 2944
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 3354
type: B, layer: 1, pos: 3354
type: B, layer: 1, pos: 2648
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 3008
type: A, layer: 1, pos: 3025
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 2338
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 2106
type: A, layer: 1, pos: 2106
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2479
type: A, layer: 1, pos: 2479
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 840
type: B, layer: 1, pos: 2337
type: B, layer: 1, pos: 2213
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 3013
type: B, layer: 1, pos: 3013
type: A, layer: 1, pos: 2547
type: B, layer: 1, pos: 2547
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 2633
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 683
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 3305
type: A, layer: 1, pos: 3305
type: A, layer: 1, pos: 2263
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 2263
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 756
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: B, layer: 1, pos: 2232
type: A, layer: 1, pos: 2232
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 787
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
type: B, layer: 1, pos: 330

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3526

## Relational analysis of NS_A1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0162019, upper bound: 0.0164098
time: 130.31 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0162018, upper bound: 0.0165244
time: 8.23 seconds

## BFS NS instance: NS_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -3.5748596, -2.7687316, -3.5749087, -2.7687316, -0.4738352, 0.4741963
1: -5.3830595, -4.0880818, -5.3830595, -4.0879474, -0.4202533, 0.4154601
2: -0.5318988, -0.3178264, -0.5319036, -0.3178236, -0.0836257, 0.0926954
3: -1.0287163, -0.6437089, -1.0287172, -0.6435572, -0.1139404, 0.1140024
4: -0.5959505, -0.0644430, -0.5960517, -0.0644412, -0.1386176, 0.1449474
5: -0.6392280, -0.2499310, -0.6392307, -0.2498779, -0.1526934, 0.1556698
6: -1.9660894, -1.2746171, -1.9660931, -1.2743422, -0.1111065, 0.1178803
7: 0.5705532, 0.9923096, 0.5702824, 0.9923112, -0.0573929, 0.0505576
8: -5.5619855, -4.4832869, -5.5620513, -4.4832745, -0.3915246, 0.3833622
9: -4.6616063, -3.6496890, -4.6616325, -3.6494477, -0.4078372, 0.3897544

Time for backsubstitution: 6.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 3526
type: B, layer: 1, pos: 3526
type: A, layer: 1, pos: 3543
type: B, layer: 1, pos: 3543
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 3542
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 3095
type: A, layer: 1, pos: 3095
type: B, layer: 1, pos: 3532
type: A, layer: 1, pos: 3532
type: A, layer: 1, pos: 3549
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 3272
type: A, layer: 1, pos: 3272
type: B, layer: 1, pos: 2650
type: A, layer: 1, pos: 2650
type: A, layer: 1, pos: 601
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 2868
type: B, layer: 1, pos: 2868
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 3287
type: B, layer: 1, pos: 2869
type: A, layer: 1, pos: 2869
type: A, layer: 1, pos: 2870
type: B, layer: 1, pos: 2870
type: A, layer: 1, pos: 2875
type: B, layer: 1, pos: 2875
type: A, layer: 1, pos: 2867
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 236
type: A, layer: 1, pos: 2158
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 3527
type: B, layer: 1, pos: 2874
type: A, layer: 1, pos: 2874
type: B, layer: 1, pos: 3499
type: A, layer: 1, pos: 2188
type: B, layer: 1, pos: 2188
type: B, layer: 1, pos: 2230
type: A, layer: 1, pos: 2230
type: A, layer: 1, pos: 3437
type: A, layer: 1, pos: 2203
type: B, layer: 1, pos: 2203
type: A, layer: 1, pos: 2577
type: B, layer: 1, pos: 2577
type: B, layer: 1, pos: 2866
type: A, layer: 1, pos: 2866
type: A, layer: 1, pos: 3100
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 172
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 3100
type: A, layer: 1, pos: 2228
type: A, layer: 1, pos: 2662
type: B, layer: 1, pos: 2662
type: A, layer: 1, pos: 3387
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 3289
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 2532
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2422
type: B, layer: 1, pos: 2422
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 2142
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 3227
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 2633
type: B, layer: 1, pos: 2513
type: A, layer: 1, pos: 2513
type: B, layer: 1, pos: 2517
type: A, layer: 1, pos: 2517
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2304
type: A, layer: 1, pos: 2303
type: B, layer: 1, pos: 2303
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 807
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 100
type: A, layer: 1, pos: 100
type: B, layer: 1, pos: 2499
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2484
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 2301
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2302
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 2873
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 808
type: B, layer: 1, pos: 808
type: B, layer: 1, pos: 2604
type: A, layer: 1, pos: 2604
type: B, layer: 1, pos: 2975
type: A, layer: 1, pos: 2975
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 840
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3480
type: B, layer: 1, pos: 3480
type: B, layer: 1, pos: 2498
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 3024
type: A, layer: 1, pos: 2062
type: B, layer: 1, pos: 2062
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 2087
type: B, layer: 1, pos: 2087
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2951
type: A, layer: 1, pos: 3510
type: B, layer: 1, pos: 3510
type: A, layer: 1, pos: 2073
type: B, layer: 1, pos: 2073
type: B, layer: 1, pos: 2980
type: A, layer: 1, pos: 2808
type: A, layer: 1, pos: 2980
type: B, layer: 1, pos: 2808
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: B, layer: 1, pos: 225
type: A, layer: 1, pos: 225
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2949
type: A, layer: 1, pos: 2949
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2483
type: A, layer: 1, pos: 2838
type: B, layer: 1, pos: 2483
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 2057
type: A, layer: 1, pos: 2270
type: B, layer: 1, pos: 2270
type: A, layer: 1, pos: 2514
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 2977
type: A, layer: 1, pos: 2977
type: B, layer: 1, pos: 3290
type: A, layer: 1, pos: 3290
type: A, layer: 1, pos: 2844
type: B, layer: 1, pos: 2844
type: A, layer: 1, pos: 2959
type: B, layer: 1, pos: 2959
type: A, layer: 1, pos: 2945
type: B, layer: 1, pos: 2945
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 2976
type: A, layer: 1, pos: 2976
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 3525
type: B, layer: 1, pos: 3525
type: B, layer: 1, pos: 2938
type: A, layer: 1, pos: 2938
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2105
type: A, layer: 1, pos: 2944
type: B, layer: 1, pos: 2944
type: B, layer: 1, pos: 2333
type: A, layer: 1, pos: 2333
type: B, layer: 1, pos: 2138
type: A, layer: 1, pos: 2138
type: B, layer: 1, pos: 2858
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2547
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: A, layer: 1, pos: 2053
type: B, layer: 1, pos: 2053
type: A, layer: 1, pos: 2283
type: B, layer: 1, pos: 330
type: A, layer: 1, pos: 2858
type: A, layer: 1, pos: 2213
type: B, layer: 1, pos: 2283
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2349
type: A, layer: 1, pos: 2338
type: B, layer: 1, pos: 2338
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 711
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 3008
type: A, layer: 1, pos: 512
type: A, layer: 1, pos: 3354
type: B, layer: 1, pos: 3354
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2106
type: B, layer: 1, pos: 2106
type: B, layer: 1, pos: 2479
type: A, layer: 1, pos: 2479
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: B, layer: 1, pos: 2213
type: A, layer: 1, pos: 3305
type: B, layer: 1, pos: 3305
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 648
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 3013
type: B, layer: 1, pos: 3013
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 2232
type: A, layer: 1, pos: 330
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 657
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 2263
type: B, layer: 1, pos: 2263
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 649
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 793
type: A, layer: 1, pos: 793
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
type: B, layer: 1, pos: 88
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
type: A, layer: 1, pos: 2232

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 3526

## Relational analysis of NS_A1_B2_A2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0162197, upper bound: 0.0165210
time: 106.89 seconds

## Relational analysis of NS_A1_B2_A2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0163269, upper bound: 0.0165248
time: 33.33 seconds

## BFS NS instance: NS_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -3.5740569, -2.7684793, -3.5733261, -2.7696738, -0.4728112, 0.4735441
1: -5.3849587, -4.0870357, -5.3830633, -4.0870214, -0.4231689, 0.4155422
2: -0.5312270, -0.3178103, -0.5310850, -0.3185774, -0.0826020, 0.0933560
3: -1.0308716, -0.6429965, -1.0282300, -0.6433865, -0.1168893, 0.1137438
4: -0.5960926, -0.0631341, -0.5958823, -0.0648950, -0.1389520, 0.1460764
5: -0.6399254, -0.2495675, -0.6390790, -0.2497263, -0.1531868, 0.1557488
6: -1.9691565, -1.2726096, -1.9655420, -1.2730476, -0.1160130, 0.1180732
7: 0.5684741, 0.9944875, 0.5705717, 0.9907264, -0.0590422, 0.0523895
8: -5.5624442, -4.4827366, -5.5620766, -4.4836545, -0.3916397, 0.3839127
9: -4.6645832, -3.6485703, -4.6611567, -3.6486657, -0.4121473, 0.3894794

Time for backsubstitution: 6.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 3526
type: A, layer: 1, pos: 3526
type: B, layer: 1, pos: 3543
type: A, layer: 1, pos: 3543
type: B, layer: 1, pos: 602
type: A, layer: 1, pos: 602
type: B, layer: 1, pos: 3542
type: A, layer: 1, pos: 3542
type: B, layer: 1, pos: 566
type: A, layer: 1, pos: 566
type: B, layer: 1, pos: 3095
type: A, layer: 1, pos: 3095
type: B, layer: 1, pos: 3532
type: A, layer: 1, pos: 3532
type: B, layer: 1, pos: 3272
type: A, layer: 1, pos: 3272
type: A, layer: 1, pos: 3549
type: B, layer: 1, pos: 3549
type: B, layer: 1, pos: 3287
type: A, layer: 1, pos: 3287
type: B, layer: 1, pos: 601
type: A, layer: 1, pos: 601
type: A, layer: 1, pos: 2869
type: B, layer: 1, pos: 2869
type: A, layer: 1, pos: 2875
type: B, layer: 1, pos: 2875
type: A, layer: 1, pos: 2870
type: B, layer: 1, pos: 2870
type: A, layer: 1, pos: 2650
type: B, layer: 1, pos: 2650
type: B, layer: 1, pos: 236
type: A, layer: 1, pos: 2868
type: B, layer: 1, pos: 2868
type: A, layer: 1, pos: 2158
type: B, layer: 1, pos: 2158
type: B, layer: 1, pos: 3527
type: A, layer: 1, pos: 2867
type: B, layer: 1, pos: 2867
type: B, layer: 1, pos: 3499
type: A, layer: 1, pos: 2188
type: B, layer: 1, pos: 2188
type: A, layer: 1, pos: 2874
type: B, layer: 1, pos: 2874
type: A, layer: 1, pos: 2577
type: B, layer: 1, pos: 2577
type: A, layer: 1, pos: 3289
type: B, layer: 1, pos: 3289
type: B, layer: 1, pos: 3387
type: A, layer: 1, pos: 3387
type: A, layer: 1, pos: 3100
type: A, layer: 1, pos: 3437
type: B, layer: 1, pos: 3100
type: B, layer: 1, pos: 3227
type: A, layer: 1, pos: 3227
type: B, layer: 1, pos: 2532
type: A, layer: 1, pos: 2532
type: A, layer: 1, pos: 2228
type: B, layer: 1, pos: 2228
type: B, layer: 1, pos: 3288
type: A, layer: 1, pos: 3288
type: B, layer: 1, pos: 2516
type: A, layer: 1, pos: 2516
type: B, layer: 1, pos: 2513
type: A, layer: 1, pos: 2513
type: A, layer: 1, pos: 2142
type: B, layer: 1, pos: 2142
type: A, layer: 1, pos: 2203
type: B, layer: 1, pos: 2203
type: A, layer: 1, pos: 2517
type: B, layer: 1, pos: 2517
type: A, layer: 1, pos: 2230
type: B, layer: 1, pos: 2230
type: A, layer: 1, pos: 2304
type: B, layer: 1, pos: 2304
type: A, layer: 1, pos: 2866
type: B, layer: 1, pos: 2866
type: B, layer: 1, pos: 100
type: A, layer: 1, pos: 100
type: B, layer: 1, pos: 2303
type: A, layer: 1, pos: 2303
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2422
type: A, layer: 1, pos: 807
type: B, layer: 1, pos: 807
type: B, layer: 1, pos: 2499
type: A, layer: 1, pos: 2499
type: A, layer: 1, pos: 2484
type: B, layer: 1, pos: 2484
type: B, layer: 1, pos: 2422
type: A, layer: 1, pos: 172
type: B, layer: 1, pos: 172
type: B, layer: 1, pos: 2301
type: A, layer: 1, pos: 2301
type: A, layer: 1, pos: 2302
type: B, layer: 1, pos: 2302
type: B, layer: 1, pos: 99
type: A, layer: 1, pos: 99
type: B, layer: 1, pos: 808
type: A, layer: 1, pos: 808
type: A, layer: 1, pos: 2662
type: B, layer: 1, pos: 2662
type: B, layer: 1, pos: 2604
type: A, layer: 1, pos: 2604
type: B, layer: 1, pos: 3480
type: A, layer: 1, pos: 3480
type: B, layer: 1, pos: 2873
type: A, layer: 1, pos: 2873
type: A, layer: 1, pos: 2275
type: B, layer: 1, pos: 2275
type: B, layer: 1, pos: 2498
type: A, layer: 1, pos: 2498
type: A, layer: 1, pos: 2975
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2975
type: A, layer: 1, pos: 2531
type: B, layer: 1, pos: 2531
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3024
type: A, layer: 1, pos: 3024
type: B, layer: 1, pos: 2062
type: A, layer: 1, pos: 2062
type: B, layer: 1, pos: 3510
type: A, layer: 1, pos: 3510
type: A, layer: 1, pos: 2808
type: B, layer: 1, pos: 2808
type: A, layer: 1, pos: 2951
type: B, layer: 1, pos: 2951
type: B, layer: 1, pos: 776
type: A, layer: 1, pos: 776
type: B, layer: 1, pos: 3290
type: A, layer: 1, pos: 3290
type: B, layer: 1, pos: 2980
type: A, layer: 1, pos: 2980
type: B, layer: 1, pos: 2559
type: A, layer: 1, pos: 2559
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 51
type: B, layer: 1, pos: 51
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 225
type: B, layer: 1, pos: 225
type: B, layer: 1, pos: 2949
type: A, layer: 1, pos: 2949
type: A, layer: 1, pos: 2483
type: B, layer: 1, pos: 2483
type: B, layer: 1, pos: 2844
type: A, layer: 1, pos: 2838
type: A, layer: 1, pos: 2844
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 2838
type: A, layer: 1, pos: 2514
type: A, layer: 1, pos: 2087
type: B, layer: 1, pos: 2514
type: B, layer: 1, pos: 2087
type: B, layer: 1, pos: 2977
type: A, layer: 1, pos: 2977
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2270
type: B, layer: 1, pos: 2270
type: A, layer: 1, pos: 2057
type: B, layer: 1, pos: 2057
type: B, layer: 1, pos: 2072
type: A, layer: 1, pos: 2072
type: A, layer: 1, pos: 2073
type: B, layer: 1, pos: 2073
type: A, layer: 1, pos: 3241
type: B, layer: 1, pos: 3241
type: B, layer: 1, pos: 2959
type: A, layer: 1, pos: 2959
type: B, layer: 1, pos: 770
type: A, layer: 1, pos: 770
type: A, layer: 1, pos: 2976
type: B, layer: 1, pos: 2976
type: A, layer: 1, pos: 2945
type: B, layer: 1, pos: 2945
type: A, layer: 1, pos: 699
type: B, layer: 1, pos: 699
type: A, layer: 1, pos: 183
type: B, layer: 1, pos: 3525
type: B, layer: 1, pos: 183
type: A, layer: 1, pos: 3525
type: A, layer: 1, pos: 2938
type: B, layer: 1, pos: 2938
type: B, layer: 1, pos: 330
type: B, layer: 1, pos: 2138
type: A, layer: 1, pos: 2138
type: A, layer: 1, pos: 2105
type: B, layer: 1, pos: 682
type: A, layer: 1, pos: 682
type: B, layer: 1, pos: 2105
type: A, layer: 1, pos: 2053
type: B, layer: 1, pos: 2053
type: B, layer: 1, pos: 2333
type: A, layer: 1, pos: 2333
type: B, layer: 1, pos: 83
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 2944
type: A, layer: 1, pos: 2858
type: B, layer: 1, pos: 2944
type: B, layer: 1, pos: 2858
type: B, layer: 1, pos: 684
type: A, layer: 1, pos: 684
type: B, layer: 1, pos: 2648
type: B, layer: 1, pos: 711
type: A, layer: 1, pos: 711
type: A, layer: 1, pos: 3354
type: B, layer: 1, pos: 2349
type: A, layer: 1, pos: 2349
type: B, layer: 1, pos: 3354
type: A, layer: 1, pos: 2283
type: A, layer: 1, pos: 2648
type: A, layer: 1, pos: 685
type: B, layer: 1, pos: 685
type: B, layer: 1, pos: 3008
type: B, layer: 1, pos: 2283
type: A, layer: 1, pos: 3008
type: B, layer: 1, pos: 3025
type: A, layer: 1, pos: 3025
type: A, layer: 1, pos: 2338
type: B, layer: 1, pos: 2338
type: A, layer: 1, pos: 2106
type: B, layer: 1, pos: 2106
type: A, layer: 1, pos: 512
type: B, layer: 1, pos: 512
type: A, layer: 1, pos: 111
type: B, layer: 1, pos: 111
type: A, layer: 1, pos: 2479
type: B, layer: 1, pos: 2479
type: B, layer: 1, pos: 648
type: A, layer: 1, pos: 648
type: A, layer: 1, pos: 840
type: B, layer: 1, pos: 840
type: A, layer: 1, pos: 2337
type: B, layer: 1, pos: 2337
type: A, layer: 1, pos: 2213
type: A, layer: 1, pos: 3013
type: B, layer: 1, pos: 2213
type: A, layer: 1, pos: 2547
type: B, layer: 1, pos: 2547
type: B, layer: 1, pos: 3013
type: A, layer: 1, pos: 748
type: B, layer: 1, pos: 748
type: A, layer: 1, pos: 2633
type: B, layer: 1, pos: 657
type: A, layer: 1, pos: 657
type: B, layer: 1, pos: 2633
type: A, layer: 1, pos: 656
type: B, layer: 1, pos: 656
type: A, layer: 1, pos: 3305
type: B, layer: 1, pos: 683
type: A, layer: 1, pos: 683
type: A, layer: 1, pos: 87
type: B, layer: 1, pos: 87
type: B, layer: 1, pos: 3305
type: A, layer: 1, pos: 2263
type: B, layer: 1, pos: 649
type: A, layer: 1, pos: 649
type: B, layer: 1, pos: 2263
type: A, layer: 1, pos: 793
type: B, layer: 1, pos: 793
type: B, layer: 1, pos: 756
type: A, layer: 1, pos: 756
type: B, layer: 1, pos: 88
type: A, layer: 1, pos: 88
type: A, layer: 1, pos: 2232
type: B, layer: 1, pos: 2232
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 787
type: A, layer: 1, pos: 787
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
type: A, layer: 1, pos: 330

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 3526

## Relational analysis of NS_A2_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0163870, upper bound: 0.0164152
time: 27.78 seconds

## Relational analysis of NS_A2_B1_A2_A2_B2

### Relational analysis result of NS_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0163873, upper bound: 0.0165215
time: 88.02 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 33.78 + 1826.39 = 1860.18 seconds
