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
execution time: IAR + RelationalAnalysis = 21.63 + 34.73 = 56.36 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.4360401, upper bound: 0.4360405

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 514
type: A, layer: 1, pos: 6142
type: A, layer: 1, pos: 6210
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5804
type: A, layer: 1, pos: 5814

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 514

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4326636, upper bound: 0.4359403
time: 5.68 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4360331, upper bound: 0.4360343
time: 3.98 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.88 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 9.88
Output dim: 8, lower bound: -0.4326636, upper bound: 0.4359403
NS_A2, status: Status.UNKNOWN, split count: 1, time: 9.88
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

Time for backsubstitution: 19.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 6142
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5804
type: B, layer: 1, pos: 5814

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 514

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4326636, upper bound: 0.4326635
time: 8.11 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4326636, upper bound: 0.4359404
time: 8.10 seconds

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

Time for backsubstitution: 21.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 514
type: B, layer: 1, pos: 6142
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5804
type: B, layer: 1, pos: 5814

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 514

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4359404, upper bound: 0.4326634
time: 5.40 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4359404, upper bound: 0.4326641
time: 4.02 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 31.28 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 31.28
Output dim: 8, lower bound: -0.4326636, upper bound: 0.4326635
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 31.28
Output dim: 8, lower bound: -0.4326636, upper bound: 0.4359404
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 31.28
Output dim: 8, lower bound: -0.4359404, upper bound: 0.4326634
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 31.28
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

Time for backsubstitution: 22.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6142
type: A, layer: 1, pos: 6210
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5804
type: A, layer: 1, pos: 5814

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 6142

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4313397, upper bound: 0.4357245
time: 3.82 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4326615, upper bound: 0.4359386
time: 5.26 seconds

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

Time for backsubstitution: 22.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6142
type: A, layer: 1, pos: 6210
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5804
type: A, layer: 1, pos: 5814

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 6142

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4346158, upper bound: 0.4324477
time: 4.13 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4359376, upper bound: 0.4326622
time: 4.39 seconds

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

Time for backsubstitution: 22.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6142
type: A, layer: 1, pos: 6210
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5804
type: A, layer: 1, pos: 5814

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 6142

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4346165, upper bound: 0.4324477
time: 5.00 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4359384, upper bound: 0.4328289
time: 3.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 31.06 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.06
Output dim: 8, lower bound: -0.4313397, upper bound: 0.4357245
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.06
Output dim: 8, lower bound: -0.4326615, upper bound: 0.4359386
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 31.06
Output dim: 8, lower bound: -0.4346158, upper bound: 0.4324477
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.06
Output dim: 8, lower bound: -0.4359376, upper bound: 0.4326622
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 31.06
Output dim: 8, lower bound: -0.4346165, upper bound: 0.4324477
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.06
Output dim: 8, lower bound: -0.4359384, upper bound: 0.4328289

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.3528657, -5.0527654, -6.3625655, -5.0444980, -0.7569757, 0.7585649
1: -14.1713600, -12.8314829, -14.1802368, -12.8220501, -0.9444299, 0.9441137
2: -7.2966623, -6.1082377, -7.3002787, -6.1061926, -0.7945013, 0.7961988
3: -3.6186566, -2.5537181, -3.6233826, -2.5508375, -0.8549089, 0.8556647
4: -8.9568806, -7.7081084, -8.9683132, -7.6973348, -0.8945780, 0.8946800
5: -4.2205048, -3.0658221, -4.2285442, -3.0529103, -0.6801977, 0.6732497
6: -4.7391214, -3.7009633, -4.7464809, -3.6895339, -0.7815251, 0.7772870
7: -12.0562487, -10.6828184, -12.0616035, -10.6777420, -0.9405117, 0.9407268
8: 6.3481770, 7.5030956, 6.3380485, 7.5106621, -0.8928776, 0.8952775
9: -3.3118043, -2.4826450, -3.3226624, -2.4744313, -0.5933020, 0.5947132

Time for backsubstitution: 21.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 6142
type: B, layer: 1, pos: 5804
type: B, layer: 1, pos: 5814

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6210

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4313302, upper bound: 0.4346718
time: 3.69 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4313382, upper bound: 0.4357221
time: 4.74 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6.3584452, -5.0469856, -6.3630910, -5.0425110, -0.7645009, 0.7589023
1: -14.1787243, -12.8229580, -14.1805744, -12.8190756, -0.9548974, 0.9437819
2: -7.2998638, -6.1068172, -7.3010979, -6.1059985, -0.7979817, 0.7989120
3: -3.6242881, -2.5518374, -3.6251798, -2.5505137, -0.8583808, 0.8617678
4: -8.9619064, -7.7031202, -8.9687986, -7.6955929, -0.9012289, 0.8973823
5: -4.2234802, -3.0588906, -4.2294950, -3.0515351, -0.6841080, 0.6838388
6: -4.7451138, -3.6939905, -4.7486110, -3.6885078, -0.7835703, 0.7870584
7: -12.0605259, -10.6780910, -12.0620127, -10.6761532, -0.9466710, 0.9442048
8: 6.3443723, 7.5055809, 6.3367219, 7.5108004, -0.8953857, 0.8988404
9: -3.3155646, -2.4798818, -3.3238428, -2.4740214, -0.5968049, 0.5989215

Time for backsubstitution: 22.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 6142
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5804
type: B, layer: 1, pos: 5814

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6210

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4326521, upper bound: 0.4348864
time: 4.62 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4326600, upper bound: 0.4359360
time: 5.65 seconds

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

Time for backsubstitution: 21.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 6142
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5804
type: B, layer: 1, pos: 5814

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6210

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4359282, upper bound: 0.4316098
time: 5.39 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4359361, upper bound: 0.4326598
time: 5.09 seconds

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

Time for backsubstitution: 21.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6210
type: B, layer: 1, pos: 6142
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5804
type: B, layer: 1, pos: 5814

Time for candidate selection: 0.17 seconds

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
time: 6.51 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 34.10 seconds
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 34.10
Output dim: 8, lower bound: -0.4313302, upper bound: 0.4346718
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 34.10
Output dim: 8, lower bound: -0.4313382, upper bound: 0.4357221
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 34.10
Output dim: 8, lower bound: -0.4326521, upper bound: 0.4348864
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 34.10
Output dim: 8, lower bound: -0.4326600, upper bound: 0.4359360
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 34.10
Output dim: 8, lower bound: -0.4359282, upper bound: 0.4316098
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 34.10
Output dim: 8, lower bound: -0.4359361, upper bound: 0.4326598
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 34.10
Output dim: 8, lower bound: -0.4360216, upper bound: 0.4317774
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 34.10
Output dim: 8, lower bound: -0.4360303, upper bound: 0.4328275

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6.3528671, -5.0527649, -6.3740559, -5.0439634, -0.7551682, 0.7671113
1: -14.1713600, -12.8314857, -14.1833096, -12.8215275, -0.9442945, 0.9471917
2: -7.2966623, -6.1082425, -7.3068118, -6.1058102, -0.7936087, 0.8024404
3: -3.6186552, -2.5537181, -3.6235418, -2.5483518, -0.8574371, 0.8554616
4: -8.9568758, -7.7081060, -8.9689426, -7.6851854, -0.9002342, 0.8928199
5: -4.2205067, -3.0658224, -4.2305193, -3.0528982, -0.6801009, 0.6747637
6: -4.7391210, -3.7009611, -4.7465725, -3.6878006, -0.7831187, 0.7771993
7: -12.0562458, -10.6828194, -12.0618477, -10.6713333, -0.9461670, 0.9406948
8: 6.3481770, 7.5030956, 6.3378830, 7.5134277, -0.8956013, 0.8952017
9: -3.3118043, -2.4826488, -3.3268812, -2.4743328, -0.5925212, 0.5981972

Time for backsubstitution: 20.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5804
type: A, layer: 1, pos: 6210
type: A, layer: 1, pos: 5814

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 904

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4269605, upper bound: 0.4356163
time: 4.36 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4313368, upper bound: 0.4357210
time: 4.15 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6.3584461, -5.0469904, -6.3745813, -5.0419779, -0.7626948, 0.7674179
1: -14.1787233, -12.8229618, -14.1836491, -12.8185501, -0.9547615, 0.9468603
2: -7.2998638, -6.1068206, -7.3076315, -6.1056108, -0.7970910, 0.8051507
3: -3.6242862, -2.5518384, -3.6253386, -2.5480313, -0.8609085, 0.8615651
4: -8.9619026, -7.7031212, -8.9694252, -7.6834412, -0.9050407, 0.8955245
5: -4.2234812, -3.0588903, -4.2314672, -3.0515237, -0.6840105, 0.6853523
6: -4.7451119, -3.6939907, -4.7487040, -3.6867731, -0.7851629, 0.7869716
7: -12.0605249, -10.6780910, -12.0622540, -10.6697464, -0.9523249, 0.9441719
8: 6.3443723, 7.5055790, 6.3365560, 7.5135617, -0.8981094, 0.8987646
9: -3.3155634, -2.4798844, -3.3280602, -2.4739232, -0.5960238, 0.6015699

Time for backsubstitution: 21.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5804
type: A, layer: 1, pos: 6210
type: A, layer: 1, pos: 5814

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 904

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4282824, upper bound: 0.4358312
time: 6.09 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4326586, upper bound: 0.4359359
time: 4.68 seconds

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

Time for backsubstitution: 20.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 6210
type: A, layer: 1, pos: 5804
type: A, layer: 1, pos: 5814

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 904

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4315506, upper bound: 0.4315045
time: 4.10 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4359268, upper bound: 0.4316088
time: 4.59 seconds

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

Time for backsubstitution: 21.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5804
type: A, layer: 1, pos: 6210
type: A, layer: 1, pos: 5814

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 904

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4315586, upper bound: 0.4325543
time: 5.66 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4359347, upper bound: 0.4326591
time: 4.75 seconds

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

Time for backsubstitution: 21.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 6210
type: A, layer: 1, pos: 5804
type: A, layer: 1, pos: 5814

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 904

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4316446, upper bound: 0.4316713
time: 7.25 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4360209, upper bound: 0.4317759
time: 4.35 seconds

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

Time for backsubstitution: 21.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 904
type: A, layer: 1, pos: 5804
type: A, layer: 1, pos: 6210
type: A, layer: 1, pos: 5814

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 904

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4316526, upper bound: 0.4327220
time: 6.46 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4360288, upper bound: 0.4328263
time: 5.52 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 34.17 seconds
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 34.17
Output dim: 8, lower bound: -0.4269605, upper bound: 0.4356163
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 34.17
Output dim: 8, lower bound: -0.4313368, upper bound: 0.4357210
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 34.17
Output dim: 8, lower bound: -0.4282824, upper bound: 0.4358312
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 34.17
Output dim: 8, lower bound: -0.4326586, upper bound: 0.4359359
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 34.17
Output dim: 8, lower bound: -0.4315506, upper bound: 0.4315045
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 34.17
Output dim: 8, lower bound: -0.4359268, upper bound: 0.4316088
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 34.17
Output dim: 8, lower bound: -0.4315586, upper bound: 0.4325543
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 34.17
Output dim: 8, lower bound: -0.4359347, upper bound: 0.4326591
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 34.17
Output dim: 8, lower bound: -0.4316446, upper bound: 0.4316713
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 34.17
Output dim: 8, lower bound: -0.4360209, upper bound: 0.4317759
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 34.17
Output dim: 8, lower bound: -0.4316526, upper bound: 0.4327220
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 34.17
Output dim: 8, lower bound: -0.4360288, upper bound: 0.4328263

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.3520975, -5.0558286, -6.3739696, -5.0447617, -0.7537830, 0.7640224
1: -14.1671305, -12.8330708, -14.1821995, -12.8217535, -0.9400382, 0.9452591
2: -7.2964401, -6.1087565, -7.3067579, -6.1059418, -0.7930417, 0.8016574
3: -3.6158547, -2.5543447, -3.6228089, -2.5483942, -0.8546243, 0.8542304
4: -8.9563160, -7.7092319, -8.9688387, -7.6854801, -0.8989382, 0.8916039
5: -4.2208242, -3.0673892, -4.2303514, -3.0533099, -0.6788540, 0.6731565
6: -4.7377243, -3.7012155, -4.7462053, -3.6878347, -0.7821574, 0.7776260
7: -12.0554390, -10.6828985, -12.0614796, -10.6715527, -0.9446940, 0.9393606
8: 6.3542128, 7.5016489, 6.3394604, 7.5132952, -0.8894672, 0.8924413
9: -3.3103528, -2.4888892, -3.3267658, -2.4759684, -0.5890393, 0.5918384

Time for backsubstitution: 22.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6142
type: B, layer: 1, pos: 5804
type: B, layer: 1, pos: 904
type: B, layer: 1, pos: 5814

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6142

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.4269605, upper bound: 0.4345090
time: 4.08 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4269605, upper bound: 0.4356164
time: 4.73 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6.3528666, -5.0527687, -6.3740559, -5.0439634, -0.7551677, 0.7670496
1: -14.1713600, -12.8314848, -14.1833096, -12.8215275, -0.9429832, 0.9471917
2: -7.2966399, -6.1082411, -7.3068118, -6.1058102, -0.7935610, 0.8024404
3: -3.6186533, -2.5537200, -3.6235418, -2.5483518, -0.8556700, 0.8554626
4: -8.9568748, -7.7081070, -8.9689426, -7.6851854, -0.9001112, 0.8926482
5: -4.2205057, -3.0658226, -4.2305193, -3.0528982, -0.6814227, 0.6747634
6: -4.7391195, -3.7009616, -4.7465725, -3.6878006, -0.7860565, 0.7771983
7: -12.0562468, -10.6828184, -12.0618477, -10.6713333, -0.9461660, 0.9417982
8: 6.3481793, 7.5030947, 6.3378830, 7.5134277, -0.8952341, 0.8952007
9: -3.3118050, -2.4826508, -3.3268812, -2.4743328, -0.5925214, 0.5940595

Time for backsubstitution: 22.04 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.36 + 545.60 = 601.96 seconds
