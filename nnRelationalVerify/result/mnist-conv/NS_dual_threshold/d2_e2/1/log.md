## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.4356040599


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.3630953, -5.0424914, -6.3630953, -5.0424914, -0.7693038, 0.7693033)
1: (-14.1805782, -12.8190308, -14.1805782, -12.8190308, -0.9576340, 0.9576340)
2: (-7.3011007, -6.1059828, -7.3011007, -6.1059828, -0.8018274, 0.8018272)
3: (-3.6252294, -2.5504999, -3.6252294, -2.5504999, -0.8629718, 0.8629718)
4: (-8.9688091, -7.6955185, -8.9688091, -7.6955185, -0.9078436, 0.9078436)
5: (-4.2295380, -3.0515246, -4.2295380, -3.0515246, -0.6916952, 0.6916952)
6: (-4.7486496, -3.6884871, -4.7486496, -3.6884871, -0.7937036, 0.7937036)
7: (-12.0620222, -10.6761436, -12.0620222, -10.6761436, -0.9489627, 0.9489627)
8: (6.3367090, 7.5108032, 6.3367090, 7.5108032, -0.9051771, 0.9051766)
9: (-3.3238935, -2.4740212, -3.3238935, -2.4740212, -0.6065361, 0.6065361)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.37 + 33.54 = 56.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.4360401, upper bound: 0.4360405

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 514
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 6142
type: B, layer: 1, pos: 6142
type: A, layer: 1, pos: 6210
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5804
type: B, layer: 1, pos: 5804
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 514

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4326636, upper bound: 0.4359403
time: 5.63 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4360331, upper bound: 0.4360343
time: 4.07 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.92 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 9.92
Output dim: 8, lower bound: -0.4326636, upper bound: 0.4359403
NS_A2, status: Status.UNKNOWN, split count: 1, time: 9.92
Output dim: 8, lower bound: -0.4360331, upper bound: 0.4360343

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -6.3584476, -5.0469832, -6.3614559, -5.0426722, -0.7645020, 0.7632353
1: -14.1787233, -12.8229570, -14.1805601, -12.8203173, -0.9535518, 0.9539061
2: -7.2998662, -6.1068139, -7.3007679, -6.1061988, -0.7996168, 0.7999578
3: -3.6242933, -2.5518365, -3.6248989, -2.5508037, -0.8606782, 0.8601542
4: -8.9619083, -7.7031169, -8.9684820, -7.6981997, -0.8986368, 0.9000854
5: -4.2234831, -3.0588865, -4.2294087, -3.0541232, -0.6831048, 0.6842475
6: -4.7451200, -3.6939902, -4.7485862, -3.6903987, -0.7865090, 0.7883883
7: -12.0605278, -10.6780901, -12.0618019, -10.6767788, -0.9460821, 0.9469161
8: 6.3443689, 7.5055804, 6.3392711, 7.5106692, -0.8973942, 0.8960075
9: -3.3155668, -2.4798808, -3.3210704, -2.4740908, -0.5981915, 0.5970216

Time for backsubstitution: 21.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 6142
type: B, layer: 1, pos: 6142
type: B, layer: 1, pos: 6210
type: A, layer: 1, pos: 6210
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5804
type: B, layer: 1, pos: 5804
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 514

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4326636, upper bound: 0.4326635
time: 8.72 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4326636, upper bound: 0.4359404
time: 8.05 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -6.3630939, -5.0424891, -6.3630929, -5.0424910, -0.7642198, 0.7693021
1: -14.1805792, -12.8190327, -14.1805792, -12.8190308, -0.9567719, 0.9552026
2: -7.3010998, -6.1059861, -7.3011022, -6.1059852, -0.8034005, 0.8005452
3: -3.6252279, -2.5505013, -3.6252298, -2.5504994, -0.8611617, 0.8654761
4: -8.9688082, -7.6955266, -8.9688091, -7.6955194, -0.9078403, 0.9008474
5: -4.2295380, -3.0515325, -4.2295389, -3.0515277, -0.6916914, 0.6820514
6: -4.7486477, -3.6884949, -4.7486467, -3.6884887, -0.7923317, 0.7899227
7: -12.0620184, -10.6761465, -12.0620222, -10.6761427, -0.9508352, 0.9486690
8: 6.3367186, 7.5108047, 6.3367128, 7.5108018, -0.8978863, 0.9039783
9: -3.3238902, -2.4740222, -3.3238931, -2.4740210, -0.5981011, 0.6059356

Time for backsubstitution: 22.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 514
type: A, layer: 1, pos: 6142
type: B, layer: 1, pos: 6142
type: A, layer: 1, pos: 6210
type: B, layer: 1, pos: 6210
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5804
type: A, layer: 1, pos: 5804
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 514

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4359404, upper bound: 0.4326634
time: 5.23 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4359404, upper bound: 0.4326641
time: 3.96 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 31.65 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 31.65
Output dim: 8, lower bound: -0.4326636, upper bound: 0.4326635
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 31.65
Output dim: 8, lower bound: -0.4326636, upper bound: 0.4359404
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 31.65
Output dim: 8, lower bound: -0.4359404, upper bound: 0.4326634
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 31.65
Output dim: 8, lower bound: -0.4359404, upper bound: 0.4326641

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -6.3584476, -5.0469832, -6.3630939, -5.0425100, -0.7646174, 0.7648754
1: -14.1787233, -12.8229570, -14.1805763, -12.8190756, -0.9548979, 0.9530563
2: -7.2998662, -6.1068139, -7.3010993, -6.1060004, -0.7987590, 0.7989130
3: -3.6242933, -2.5518365, -3.6251798, -2.5505147, -0.8587413, 0.8589463
4: -8.9619083, -7.7031169, -8.9687977, -7.6955910, -0.9012289, 0.9002290
5: -4.2234831, -3.0588865, -4.2294941, -3.0515325, -0.6856954, 0.6843295
6: -4.7451200, -3.6939902, -4.7486134, -3.6885052, -0.7887993, 0.7870622
7: -12.0605278, -10.6780901, -12.0620127, -10.6761522, -0.9466715, 0.9460239
8: 6.3443689, 7.5055804, 6.3367195, 7.5107999, -0.8963003, 0.8988414
9: -3.3155668, -2.4798808, -3.3238447, -2.4740219, -0.5976443, 0.6000462

Time for backsubstitution: 21.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6142
type: A, layer: 1, pos: 6142
type: A, layer: 1, pos: 6210
type: B, layer: 1, pos: 6210
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5804
type: A, layer: 1, pos: 5804
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6142

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4324473, upper bound: 0.4346170
time: 3.84 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4326618, upper bound: 0.4359384
time: 5.39 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -6.3630939, -5.0424891, -6.3584476, -5.0469832, -0.7648816, 0.7646434
1: -14.1805792, -12.8190327, -14.1787233, -12.8229570, -0.9530578, 0.9549189
2: -7.3010998, -6.1059861, -7.2998662, -6.1068139, -0.7989144, 0.7987723
3: -3.6252279, -2.5505013, -3.6242933, -2.5518365, -0.8590279, 0.8587432
4: -8.9688082, -7.6955266, -8.9619083, -7.7031169, -0.9002333, 0.9013252
5: -4.2295380, -3.0515325, -4.2234831, -3.0588865, -0.6843400, 0.6857002
6: -4.7486477, -3.6884949, -4.7451200, -3.6939902, -0.7870994, 0.7888050
7: -12.0620184, -10.6761465, -12.0605278, -10.6780901, -0.9460335, 0.9467235
8: 6.3367186, 7.5108047, 6.3443689, 7.5055804, -0.8988552, 0.8963051
9: -3.3238902, -2.4740222, -3.3155668, -2.4798808, -0.6000824, 0.5976562

Time for backsubstitution: 22.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6142
type: B, layer: 1, pos: 6142
type: B, layer: 1, pos: 6210
type: A, layer: 1, pos: 6210
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5804
type: B, layer: 1, pos: 5804
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 6142

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4346158, upper bound: 0.4324477
time: 4.28 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4359376, upper bound: 0.4326622
time: 4.44 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -6.3630939, -5.0424891, -6.3630939, -5.0424891, -0.7642193, 0.7642193
1: -14.1805792, -12.8190327, -14.1805792, -12.8190327, -0.9552011, 0.9552011
2: -7.3010998, -6.1059861, -7.3010998, -6.1059861, -0.8033991, 0.8033991
3: -3.6252279, -2.5505013, -3.6252279, -2.5505013, -0.8654714, 0.8654714
4: -8.9688082, -7.6955266, -8.9688082, -7.6955266, -0.9008484, 0.9008484
5: -4.2295380, -3.0515325, -4.2295380, -3.0515325, -0.6820514, 0.6820517
6: -4.7486477, -3.6884949, -4.7486477, -3.6884949, -0.7899218, 0.7899218
7: -12.0620184, -10.6761465, -12.0620184, -10.6761465, -0.9508348, 0.9508348
8: 6.3367186, 7.5108047, 6.3367186, 7.5108047, -0.8978853, 0.8978853
9: -3.3238902, -2.4740222, -3.3238902, -2.4740222, -0.5981002, 0.5981002

Time for backsubstitution: 21.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6142
type: B, layer: 1, pos: 6142
type: A, layer: 1, pos: 6210
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5804
type: A, layer: 1, pos: 5804
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6142

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4346165, upper bound: 0.4324477
time: 4.61 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4359384, upper bound: 0.4328289
time: 3.61 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 30.30 seconds
NS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 30.30
Output dim: 8, lower bound: -0.4324473, upper bound: 0.4346170
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 30.30
Output dim: 8, lower bound: -0.4326618, upper bound: 0.4359384
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 30.30
Output dim: 8, lower bound: -0.4346158, upper bound: 0.4324477
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.30
Output dim: 8, lower bound: -0.4359376, upper bound: 0.4326622
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 30.30
Output dim: 8, lower bound: -0.4346165, upper bound: 0.4324477
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.30
Output dim: 8, lower bound: -0.4359384, upper bound: 0.4328289

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -6.3584461, -5.0469837, -6.3630891, -5.0425138, -0.7586424, 0.7647564
1: -14.1787224, -12.8229542, -14.1805744, -12.8190746, -0.9456239, 0.9530544
2: -7.2998648, -6.1068163, -7.3010950, -6.1059995, -0.7987576, 0.7981343
3: -3.6242919, -2.5518365, -3.6251750, -2.5505166, -0.8615623, 0.8585858
4: -8.9619045, -7.7031188, -8.9687967, -7.6955929, -0.8983822, 0.9002275
5: -4.2234821, -3.0588875, -4.2294912, -3.0515373, -0.6852052, 0.6827404
6: -4.7451191, -3.6939919, -4.7486091, -3.6885076, -0.7887955, 0.7818336
7: -12.0605268, -10.6780910, -12.0620089, -10.6761560, -0.9448538, 0.9460211
8: 6.3443708, 7.5055819, 6.3367252, 7.5107989, -0.8962994, 0.8979282
9: -3.3155653, -2.4798806, -3.3238416, -2.4740219, -0.5965211, 0.5991988

Time for backsubstitution: 21.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6210
type: B, layer: 1, pos: 6210
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 5804
type: B, layer: 1, pos: 5804
type: A, layer: 1, pos: 6142
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6210

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4316098, upper bound: 0.4359286
time: 4.39 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4326600, upper bound: 0.4359366
time: 5.66 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6.3630891, -5.0424948, -6.3584461, -5.0469837, -0.7647629, 0.7586679
1: -14.1805763, -12.8190346, -14.1787224, -12.8229542, -0.9530554, 0.9456453
2: -7.3011007, -6.1059852, -7.2998648, -6.1068163, -0.7981358, 0.7987707
3: -3.6252236, -2.5505023, -3.6242919, -2.5518365, -0.8586674, 0.8615651
4: -8.9688072, -7.6955299, -8.9619045, -7.7031188, -0.9002314, 0.8984799
5: -4.2295337, -3.0515373, -4.2234821, -3.0588875, -0.6827505, 0.6852105
6: -4.7486410, -3.6884971, -4.7451191, -3.6939919, -0.7818718, 0.7888002
7: -12.0620203, -10.6761465, -12.0605268, -10.6780910, -0.9460301, 0.9449053
8: 6.3367224, 7.5108047, 6.3443708, 7.5055819, -0.8979416, 0.8963051
9: -3.3238871, -2.4740219, -3.3155653, -2.4798806, -0.5992346, 0.5965328

Time for backsubstitution: 22.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6210
type: A, layer: 1, pos: 6210
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5804
type: A, layer: 1, pos: 5804
type: B, layer: 1, pos: 6142
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 6210

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4359282, upper bound: 0.4316098
time: 5.27 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4359361, upper bound: 0.4326598
time: 4.92 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6.3630891, -5.0424948, -6.3630910, -5.0424910, -0.7641008, 0.7582431
1: -14.1805763, -12.8190346, -14.1805763, -12.8190336, -0.9552002, 0.9459267
2: -7.3011007, -6.1059852, -7.3010998, -6.1059852, -0.8026195, 0.8033977
3: -3.6252236, -2.5505023, -3.6252265, -2.5505037, -0.8651099, 0.8682938
4: -8.9688072, -7.6955299, -8.9688082, -7.6955276, -0.9008455, 0.8979998
5: -4.2295337, -3.0515373, -4.2295365, -3.0515351, -0.6804626, 0.6815619
6: -4.7486410, -3.6884971, -4.7486453, -3.6884952, -0.7846932, 0.7899170
7: -12.0620203, -10.6761465, -12.0620222, -10.6761417, -0.9508328, 0.9490161
8: 6.3367224, 7.5108047, 6.3367186, 7.5108027, -0.8969727, 0.8978853
9: -3.3238871, -2.4740219, -3.3238883, -2.4740219, -0.5972526, 0.5969770

Time for backsubstitution: 21.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6210
type: A, layer: 1, pos: 6210
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5804
type: A, layer: 1, pos: 5804
type: B, layer: 1, pos: 6142
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6210

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4360216, upper bound: 0.4317774
time: 5.64 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4360303, upper bound: 0.4328275
time: 6.56 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 34.35 seconds
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 34.35
Output dim: 8, lower bound: -0.4316098, upper bound: 0.4359286
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 34.35
Output dim: 8, lower bound: -0.4326600, upper bound: 0.4359366
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 34.35
Output dim: 8, lower bound: -0.4359282, upper bound: 0.4316098
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 34.35
Output dim: 8, lower bound: -0.4359361, upper bound: 0.4326598
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 34.35
Output dim: 8, lower bound: -0.4360216, upper bound: 0.4317774
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 34.35
Output dim: 8, lower bound: -0.4360303, upper bound: 0.4328275

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -6.3584132, -5.0547285, -6.3630848, -5.0443220, -0.7565131, 0.7568858
1: -14.1787138, -12.8250198, -14.1805735, -12.8195562, -0.9451075, 0.9508834
2: -7.2997398, -6.1113639, -7.3010650, -6.1070614, -0.7973595, 0.7934921
3: -3.6220760, -2.5519843, -3.6246576, -2.5505490, -0.8592429, 0.8580036
4: -8.9535255, -7.7032499, -8.9668407, -7.6956224, -0.8899379, 0.8978934
5: -4.2233062, -3.0601025, -4.2294517, -3.0518205, -0.6843231, 0.6811764
6: -4.7442675, -3.6943824, -4.7484093, -3.6885991, -0.7878103, 0.7812347
7: -12.0568314, -10.6781511, -12.0611486, -10.6761703, -0.9402952, 0.9442425
8: 6.3459353, 7.5054541, 6.3370895, 7.5107703, -0.8945780, 0.8974180
9: -3.3149347, -2.4827003, -3.3236938, -2.4746790, -0.5946133, 0.5960841

Time for backsubstitution: 21.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 5804
type: B, layer: 1, pos: 5804
type: B, layer: 1, pos: 6210
type: A, layer: 1, pos: 6142
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 904

## Relational analysis of NS_A1_B2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4272321, upper bound: 0.4358229
time: 4.04 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2

### Relational analysis result of NS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4316084, upper bound: 0.4359276
time: 3.86 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -6.3699379, -5.0464454, -6.3630900, -5.0425158, -0.7659292, 0.7629533
1: -14.1817999, -12.8224335, -14.1805744, -12.8190775, -0.9487023, 0.9529161
2: -7.3063960, -6.1064324, -7.3010955, -6.1060038, -0.8049903, 0.7972391
3: -3.6244488, -2.5493507, -3.6251740, -2.5505176, -0.8613582, 0.8611145
4: -8.9625368, -7.6909695, -8.9687920, -7.6955948, -0.8965240, 0.9026284
5: -4.2254362, -3.0588768, -4.2294931, -3.0515370, -0.6867154, 0.6826432
6: -4.7452106, -3.6922543, -4.7486076, -3.6885095, -0.7887073, 0.7834287
7: -12.0607681, -10.6716805, -12.0620089, -10.6761560, -0.9448214, 0.9516745
8: 6.3442068, 7.5083408, 6.3367271, 7.5107985, -0.8962164, 0.9006457
9: -3.3197801, -2.4797807, -3.3238423, -2.4740243, -0.5976264, 0.5984182

Time for backsubstitution: 21.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5804
type: A, layer: 1, pos: 5804
type: A, layer: 1, pos: 6142
type: B, layer: 1, pos: 6210
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 904

## Relational analysis of NS_A1_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4282824, upper bound: 0.4358309
time: 4.33 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4326586, upper bound: 0.4359270
time: 5.99 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -6.3630848, -5.0442996, -6.3584132, -5.0547285, -0.7568924, 0.7565389
1: -14.1805735, -12.8195162, -14.1787138, -12.8250198, -0.9508843, 0.9451284
2: -7.3010683, -6.1070476, -7.2997398, -6.1113639, -0.7934937, 0.7973731
3: -3.6247044, -2.5505362, -3.6220760, -2.5519843, -0.8580852, 0.8592448
4: -8.9668503, -7.6955581, -8.9535255, -7.7032499, -0.8978972, 0.8900342
5: -4.2294950, -3.0518205, -4.2233062, -3.0601025, -0.6811862, 0.6843286
6: -4.7484450, -3.6885853, -4.7442675, -3.6943824, -0.7812710, 0.7878165
7: -12.0611582, -10.6761608, -12.0568314, -10.6781511, -0.9442511, 0.9403472
8: 6.3370862, 7.5107732, 6.3459353, 7.5054541, -0.8974333, 0.8945847
9: -3.3237391, -2.4746797, -3.3149347, -2.4827003, -0.5961194, 0.5946250

Time for backsubstitution: 22.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5804
type: A, layer: 1, pos: 5804
type: A, layer: 1, pos: 6210
type: B, layer: 1, pos: 6142
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 904

## Relational analysis of NS_A2_B1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4358224, upper bound: 0.4272326
time: 4.38 seconds

## Relational analysis of NS_A2_B1_A2_B1_B2

### Relational analysis result of NS_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4359271, upper bound: 0.4316084
time: 5.50 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -6.3630910, -5.0424957, -6.3699379, -5.0464454, -0.7629602, 0.7659776
1: -14.1805763, -12.8190384, -14.1817999, -12.8224335, -0.9529161, 0.9487238
2: -7.3011007, -6.1059904, -7.3063960, -6.1064324, -0.7972403, 0.8050039
3: -3.6252217, -2.5505023, -3.6244488, -2.5493507, -0.8611956, 0.8613601
4: -8.9688044, -7.6955304, -8.9625368, -7.6909695, -0.9026313, 0.8966217
5: -4.2295341, -3.0515370, -4.2254362, -3.0588768, -0.6826532, 0.6867208
6: -4.7486410, -3.6884952, -4.7452106, -3.6922543, -0.7834654, 0.7887120
7: -12.0620184, -10.6761446, -12.0607681, -10.6716805, -0.9516830, 0.9448724
8: 6.3367224, 7.5108042, 6.3442068, 7.5083408, -0.9006596, 0.8962212
9: -3.3238852, -2.4740255, -3.3197801, -2.4797807, -0.5984542, 0.5976399

Time for backsubstitution: 22.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5804
type: B, layer: 1, pos: 5804
type: B, layer: 1, pos: 6142
type: A, layer: 1, pos: 6210
type: B, layer: 1, pos: 5814
type: A, layer: 1, pos: 5814

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 904

## Relational analysis of NS_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4358304, upper bound: 0.4282829
time: 3.87 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4359351, upper bound: 0.4326585
time: 4.56 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -6.3630848, -5.0442996, -6.3630567, -5.0502353, -0.7562287, 0.7561142
1: -14.1805735, -12.8195162, -14.1805639, -12.8210945, -0.9530268, 0.9454083
2: -7.3010683, -6.1070476, -7.3009758, -6.1105309, -0.7979803, 0.8019986
3: -3.6247044, -2.5505362, -3.6230135, -2.5506458, -0.8645282, 0.8659744
4: -8.9668503, -7.6955581, -8.9604216, -7.6956587, -0.8985109, 0.8895512
5: -4.2294950, -3.0518205, -4.2293587, -3.0527496, -0.6788983, 0.6806803
6: -4.7484450, -3.6885853, -4.7477946, -3.6888878, -0.7840939, 0.7889342
7: -12.0611582, -10.6761608, -12.0583277, -10.6762047, -0.9490533, 0.9444585
8: 6.3370862, 7.5107732, 6.3382792, 7.5106788, -0.8964605, 0.8961635
9: -3.3237391, -2.4746797, -3.3232574, -2.4768419, -0.5941372, 0.5950675

Time for backsubstitution: 21.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 5804
type: A, layer: 1, pos: 5804
type: A, layer: 1, pos: 6210
type: B, layer: 1, pos: 6142
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 904

## Relational analysis of NS_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4359165, upper bound: 0.4273998
time: 3.74 seconds

## Relational analysis of NS_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4360212, upper bound: 0.4317756
time: 4.83 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6.3630910, -5.0424957, -6.3745813, -5.0419555, -0.7622957, 0.7694025
1: -14.1805763, -12.8190384, -14.1836529, -12.8185110, -0.9550591, 0.9490051
2: -7.3011007, -6.1059904, -7.3076363, -6.1055994, -0.8017287, 0.8096309
3: -3.6252217, -2.5505023, -3.6253853, -2.5480165, -0.8676400, 0.8680902
4: -8.9688044, -7.6955304, -8.9694366, -7.6833773, -0.9102545, 0.8961425
5: -4.2295341, -3.0515370, -4.2315092, -3.0515237, -0.6803658, 0.6830745
6: -4.7486410, -3.6884952, -4.7487369, -3.6867642, -0.7862864, 0.7898283
7: -12.0620184, -10.6761446, -12.0622663, -10.6697369, -0.9564886, 0.9489837
8: 6.3367224, 7.5108042, 6.3365536, 7.5135660, -0.8996925, 0.8978028
9: -3.3238852, -2.4740255, -3.3281035, -2.4739232, -0.5964727, 0.6004508

Time for backsubstitution: 22.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 904
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5804
type: A, layer: 1, pos: 5804
type: B, layer: 1, pos: 6142
type: A, layer: 1, pos: 6210
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 904

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4316526, upper bound: 0.4327220
time: 6.45 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4360288, upper bound: 0.4328263
time: 5.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 34.44 seconds
NS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 34.44
Output dim: 8, lower bound: -0.4272321, upper bound: 0.4358229
NS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 34.44
Output dim: 8, lower bound: -0.4316084, upper bound: 0.4359276
NS_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 34.44
Output dim: 8, lower bound: -0.4282824, upper bound: 0.4358309
NS_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 34.44
Output dim: 8, lower bound: -0.4326586, upper bound: 0.4359270
NS_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 34.44
Output dim: 8, lower bound: -0.4358224, upper bound: 0.4272326
NS_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 34.44
Output dim: 8, lower bound: -0.4359271, upper bound: 0.4316084
NS_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 34.44
Output dim: 8, lower bound: -0.4358304, upper bound: 0.4282829
NS_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 34.44
Output dim: 8, lower bound: -0.4359351, upper bound: 0.4326585
NS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 34.44
Output dim: 8, lower bound: -0.4359165, upper bound: 0.4273998
NS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 34.44
Output dim: 8, lower bound: -0.4360212, upper bound: 0.4317756
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 34.44
Output dim: 8, lower bound: -0.4316526, upper bound: 0.4327220
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 34.44
Output dim: 8, lower bound: -0.4360288, upper bound: 0.4328263

## BFS NS instance: NS_A1_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -6.3576469, -5.0577888, -6.3629956, -5.0451198, -0.7551403, 0.7537830
1: -14.1744843, -12.8266058, -14.1794624, -12.8197880, -0.9408507, 0.9489503
2: -7.2995157, -6.1118784, -7.3010116, -6.1071897, -0.7967920, 0.7927120
3: -3.6192770, -2.5526085, -3.6239233, -2.5505924, -0.8564296, 0.8567734
4: -8.9529638, -7.7043743, -8.9667397, -7.6959209, -0.8891406, 0.8966770
5: -4.2236309, -3.0616689, -4.2293086, -3.0522316, -0.6830769, 0.6795723
6: -4.7428665, -3.6946332, -4.7480416, -3.6886327, -0.7868505, 0.7816586
7: -12.0560112, -10.6782293, -12.0607805, -10.6763906, -0.9388256, 0.9429073
8: 6.3519669, 7.5040226, 6.3386660, 7.5106401, -0.8884506, 0.8946700
9: -3.3134861, -2.4889412, -3.3235779, -2.4763157, -0.5897064, 0.5897272

Time for backsubstitution: 22.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5804
type: A, layer: 1, pos: 5804
type: B, layer: 1, pos: 6210
type: A, layer: 1, pos: 6142
type: B, layer: 1, pos: 904
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5804

## Relational analysis of NS_A1_B2_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4272310, upper bound: 0.4351419
time: 5.72 seconds

## Relational analysis of NS_A1_B2_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4272310, upper bound: 0.4358218
time: 4.68 seconds

## BFS NS instance: NS_A1_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -6.3584146, -5.0547280, -6.3630848, -5.0443220, -0.7565131, 0.7568231
1: -14.1787128, -12.8250217, -14.1805735, -12.8195562, -0.9437957, 0.9508829
2: -7.2997231, -6.1113658, -7.3010650, -6.1070614, -0.7973113, 0.7934916
3: -3.6220756, -2.5519834, -3.6246576, -2.5505490, -0.8574758, 0.8580041
4: -8.9535227, -7.7032485, -8.9668407, -7.6956224, -0.8899374, 0.8977203
5: -4.2233062, -3.0601027, -4.2294517, -3.0518205, -0.6856449, 0.6811759
6: -4.7442646, -3.6943824, -4.7484093, -3.6885991, -0.7907495, 0.7812338
7: -12.0568323, -10.6781540, -12.0611486, -10.6761703, -0.9402957, 0.9453459
8: 6.3459334, 7.5054550, 6.3370895, 7.5107703, -0.8942127, 0.8974171
9: -3.3149343, -2.4827023, -3.3236938, -2.4746790, -0.5945740, 0.5919452

Time for backsubstitution: 21.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5804
type: A, layer: 1, pos: 5804
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 6210
type: A, layer: 1, pos: 6142
type: A, layer: 1, pos: 5814
type: B, layer: 1, pos: 5814

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 5804

## Relational analysis of NS_A1_B2_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4316075, upper bound: 0.4352472
time: 3.88 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4316074, upper bound: 0.4359265
time: 4.20 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -6.3691597, -5.0495129, -6.3630033, -5.0433187, -0.7633412, 0.7598503
1: -14.1775684, -12.8240089, -14.1794672, -12.8193092, -0.9444461, 0.9509826
2: -7.3061762, -6.1069479, -7.3010421, -6.1061354, -0.8044219, 0.7964592
3: -3.6216478, -2.5499730, -3.6244397, -2.5505590, -0.8585443, 0.8598828
4: -8.9619761, -7.6920929, -8.9686909, -7.6958904, -0.8957272, 0.9014118
5: -4.2257690, -3.0604434, -4.2293501, -3.0519493, -0.6854548, 0.6810403
6: -4.7438045, -3.6925089, -4.7482405, -3.6885462, -0.7877469, 0.7838540
7: -12.0599623, -10.6717749, -12.0616426, -10.6763744, -0.9433513, 0.9503303
8: 6.3502436, 7.5068808, 6.3383021, 7.5106711, -0.8900890, 0.8978701
9: -3.3183236, -2.4860215, -3.3237243, -2.4756613, -0.5917847, 0.5920625

Time for backsubstitution: 22.00 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.91 + 546.85 = 603.76 seconds
