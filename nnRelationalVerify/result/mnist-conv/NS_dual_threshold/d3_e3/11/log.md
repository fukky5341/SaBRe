## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.441476739


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.1999397, -9.9034901, -12.1999397, -9.9034901, -1.5637841, 1.5637841)
1: (-11.3694668, -9.5723295, -11.3694668, -9.5723295, -1.1966600, 1.1966598)
2: (-7.8098774, -6.4722166, -7.8098774, -6.4722166, -0.9311540, 0.9311540)
3: (-7.3257384, -5.4926596, -7.3257384, -5.4926596, -1.4207768, 1.4207768)
4: (-3.2507381, -1.6251090, -3.2507381, -1.6251090, -1.4014397, 1.4014397)
5: (-5.6533022, -4.1650996, -5.6533022, -4.1650996, -1.2628298, 1.2628298)
6: (-16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2654819, 1.2654817)
7: (-4.3670073, -2.6725664, -4.3670073, -2.6725664, -1.3419938, 1.3419938)
8: (-4.9663148, -3.3447104, -4.9663148, -3.3447104, -0.9227247, 0.9227247)
9: (4.7270503, 5.8195791, 4.7270503, 5.8195791, -0.7659516, 0.7659515)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.36 + 34.05 = 57.41 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.4459361, upper bound: 0.4459358

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5799
type: B, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 6166
type: A, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441482, upper bound: 0.4452840
time: 3.07 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4459356, upper bound: 0.4459378
time: 3.23 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.40 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.40
Output dim: 9, lower bound: -0.4441482, upper bound: 0.4452840
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.40
Output dim: 9, lower bound: -0.4459356, upper bound: 0.4459378

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -12.1946039, -9.9171696, -12.1980915, -9.9086761, -1.5629234, 1.5593100
1: -11.3630943, -9.5806046, -11.3671665, -9.5754185, -1.1786971, 1.1785452
2: -7.7934823, -6.4824185, -7.8041286, -6.4763622, -0.9131715, 0.9158170
3: -7.2923265, -5.5005465, -7.3122263, -5.4939098, -1.3874331, 1.4000559
4: -3.2380972, -1.6541235, -3.2480063, -1.6370251, -1.3741837, 1.3656604
5: -5.6485596, -4.1813474, -5.6521826, -4.1722336, -1.2572970, 1.2574873
6: -16.5250416, -14.3077488, -16.5345764, -14.2923384, -1.2390895, 1.2337701
7: -4.3258805, -2.6930768, -4.3500509, -2.6770880, -1.2993584, 1.3092308
8: -4.9505787, -3.3531990, -4.9597869, -3.3471761, -0.9001975, 0.9001483
9: 4.7373800, 5.8170915, 4.7311201, 5.8188386, -0.7555287, 0.7565070

Time for backsubstitution: 22.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 6166
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441470, upper bound: 0.4449301
time: 2.94 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441470, upper bound: 0.4452832
time: 3.10 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -12.1999378, -9.9034920, -12.1999378, -9.9034863, -1.5635476, 1.5593486
1: -11.3694649, -9.5723305, -11.3694668, -9.5723276, -1.2009711, 1.1966112
2: -7.8098755, -6.4722180, -7.8098783, -6.4722166, -0.9317229, 0.9311460
3: -7.3257332, -5.4926591, -7.3257384, -5.4926591, -1.4070816, 1.4207773
4: -3.2507384, -1.6251104, -3.2507391, -1.6251090, -1.4014397, 1.3904769
5: -5.6533031, -4.1651030, -5.6533031, -4.1650996, -1.2600989, 1.2629659
6: -16.5381145, -14.2816582, -16.5381165, -14.2816572, -1.2654798, 1.2506084
7: -4.3670030, -2.6725674, -4.3670082, -2.6725657, -1.3181453, 1.3419929
8: -4.9663124, -3.3447099, -4.9663153, -3.3447099, -0.9146497, 0.9214137
9: 4.7270527, 5.8195777, 4.7270503, 5.8195786, -0.7685866, 0.7644274

Time for backsubstitution: 21.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 6166
type: B, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 5799

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4459344, upper bound: 0.4455779
time: 3.35 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4459344, upper bound: 0.4459346
time: 3.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 28.08 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 28.08
Output dim: 9, lower bound: -0.4441470, upper bound: 0.4449301
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 28.08
Output dim: 9, lower bound: -0.4441470, upper bound: 0.4452832
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 28.08
Output dim: 9, lower bound: -0.4459344, upper bound: 0.4455779
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 28.08
Output dim: 9, lower bound: -0.4459344, upper bound: 0.4459346

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -12.1845436, -9.9172935, -12.1932411, -9.9087343, -1.5532327, 1.5545657
1: -11.3617325, -9.5809488, -11.3665142, -9.5755835, -1.1759605, 1.1764026
2: -7.7876320, -6.4826908, -7.8013105, -6.4764929, -0.9069681, 0.9126821
3: -7.2851486, -5.5007591, -7.3087749, -5.4940147, -1.3798933, 1.3961191
4: -3.2377484, -1.6555631, -3.2478375, -1.6377201, -1.3729401, 1.3638740
5: -5.6457996, -4.1815376, -5.6508503, -4.1723280, -1.2505169, 1.2522504
6: -16.5179596, -14.3080692, -16.5311680, -14.2924881, -1.2319751, 1.2302423
7: -4.3251495, -2.6934576, -4.3496933, -2.6772740, -1.2983413, 1.3084090
8: -4.9502516, -3.3555541, -4.9596291, -3.3483171, -0.8985455, 0.8974342
9: 4.7376595, 5.8160019, 4.7312574, 5.8183136, -0.7538502, 0.7543142

Time for backsubstitution: 21.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6166
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6166

## Relational analysis of NS_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4433477, upper bound: 0.4432076
time: 3.12 seconds

## Relational analysis of NS_A1_A1_A2

### Relational analysis result of NS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441460, upper bound: 0.4449266
time: 3.90 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -12.1966152, -9.9053020, -12.1980724, -9.9086761, -1.5596938, 1.5711975
1: -11.3642540, -9.5780659, -11.3671646, -9.5754204, -1.1800046, 1.1803238
2: -7.7941828, -6.4747610, -7.8041182, -6.4763622, -0.9108303, 0.9232900
3: -7.2938070, -5.4900126, -7.3122163, -5.4939108, -1.3849931, 1.4095886
4: -3.2409031, -1.6536164, -3.2480063, -1.6370268, -1.3768835, 1.3659155
5: -5.6495352, -4.1767631, -5.6521778, -4.1722331, -1.2589636, 1.2596483
6: -16.5265408, -14.2986183, -16.5345631, -14.2923412, -1.2366686, 1.2428405
7: -4.3276606, -2.6908708, -4.3500495, -2.6770902, -1.3003230, 1.3112369
8: -4.9537506, -3.3509359, -4.9597850, -3.3471799, -0.9033632, 0.9014351
9: 4.7350745, 5.8173761, 4.7311206, 5.8188372, -0.7572718, 0.7566366

Time for backsubstitution: 21.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6166
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6166

## Relational analysis of NS_A1_A2_A1

### Relational analysis result of NS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4433477, upper bound: 0.4435450
time: 3.15 seconds

## Relational analysis of NS_A1_A2_A2

### Relational analysis result of NS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441460, upper bound: 0.4452800
time: 4.58 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -12.1898804, -9.9036140, -12.1950874, -9.9035482, -1.5538554, 1.5545993
1: -11.3681087, -9.5726700, -11.3688116, -9.5724926, -1.1982269, 1.1944780
2: -7.8040233, -6.4724908, -7.8070583, -6.4723473, -0.9255447, 0.9280118
3: -7.3185687, -5.4928751, -7.3222885, -5.4927635, -1.3995523, 1.4168401
4: -3.2503877, -1.6265416, -3.2505679, -1.6258042, -1.4001985, 1.3886924
5: -5.6505442, -4.1653004, -5.6519699, -4.1651955, -1.2533035, 1.2577183
6: -16.5310345, -14.2819796, -16.5347061, -14.2818089, -1.2583613, 1.2470779
7: -4.3662786, -2.6729560, -4.3666496, -2.6727533, -1.3171306, 1.3411655
8: -4.9659891, -3.3470564, -4.9661584, -3.3458505, -0.9129913, 0.9186940
9: 4.7273350, 5.8184891, 4.7271872, 5.8190536, -0.7669103, 0.7622308

Time for backsubstitution: 22.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6166
type: B, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5799

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 6166

## Relational analysis of NS_A2_A1_A1

### Relational analysis result of NS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4452149, upper bound: 0.4440550
time: 3.28 seconds

## Relational analysis of NS_A2_A1_A2

### Relational analysis result of NS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4459335, upper bound: 0.4455771
time: 6.37 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -12.2019482, -9.8916187, -12.1999187, -9.9034901, -1.5603209, 1.5712352
1: -11.3706207, -9.5697794, -11.3694620, -9.5723286, -1.2022686, 1.1984141
2: -7.8105607, -6.4645600, -7.8098679, -6.4722171, -0.9293931, 0.9386218
3: -7.3272171, -5.4821281, -7.3257284, -5.4926586, -1.4046555, 1.4308925
4: -3.2535460, -1.6245790, -3.2507377, -1.6251125, -1.4041562, 1.3907304
5: -5.6542797, -4.1605072, -5.6532989, -4.1651011, -1.2617378, 1.2651193
6: -16.5396156, -14.2725277, -16.5381031, -14.2816544, -1.2630432, 1.2596810
7: -4.3687754, -2.6703329, -4.3670063, -2.6725671, -1.3191195, 1.3439982
8: -4.9694891, -3.3424582, -4.9663134, -3.3447123, -0.9178131, 0.9227196
9: 4.7247438, 5.8198614, 4.7270508, 5.8195782, -0.7703280, 0.7645564

Time for backsubstitution: 22.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6166
type: B, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5799

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6166

## Relational analysis of NS_A2_A2_A1

### Relational analysis result of NS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4452149, upper bound: 0.4443976
time: 3.33 seconds

## Relational analysis of NS_A2_A2_A2

### Relational analysis result of NS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4459335, upper bound: 0.4459330
time: 4.68 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 30.61 seconds
NS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 30.61
Output dim: 9, lower bound: -0.4433477, upper bound: 0.4432076
NS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 30.61
Output dim: 9, lower bound: -0.4441460, upper bound: 0.4449266
NS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 30.61
Output dim: 9, lower bound: -0.4433477, upper bound: 0.4435450
NS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 30.61
Output dim: 9, lower bound: -0.4441460, upper bound: 0.4452800
NS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 30.61
Output dim: 9, lower bound: -0.4452149, upper bound: 0.4440550
NS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 30.61
Output dim: 9, lower bound: -0.4459335, upper bound: 0.4455771
NS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 30.61
Output dim: 9, lower bound: -0.4452149, upper bound: 0.4443976
NS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 30.61
Output dim: 9, lower bound: -0.4459335, upper bound: 0.4459330

## BFS NS instance: NS_A1_A1_A1

### Backsubstitution after applying NS history:
0: -12.1617756, -9.9606438, -12.1904058, -9.9299412, -1.5027227, 1.5085177
1: -11.3475599, -9.6045341, -11.3646250, -9.5865612, -1.1502786, 1.1522067
2: -7.7457032, -6.4994516, -7.7780895, -6.4777403, -0.8654335, 0.8736739
3: -7.2556467, -5.5603809, -7.3037891, -5.5222540, -1.3238573, 1.3333335
4: -3.2251983, -1.6618605, -3.2436390, -1.6409879, -1.3533044, 1.3486836
5: -5.6182165, -4.2234697, -5.6450348, -4.1921268, -1.2013259, 1.2030158
6: -16.5062695, -14.3203602, -16.5273705, -14.2981186, -1.2113345, 1.2123899
7: -4.3118916, -2.7134852, -4.3430142, -2.6817577, -1.2812381, 1.2801449
8: -4.9403973, -3.3623924, -4.9574065, -3.3521962, -0.8811185, 0.8870844
9: 4.7454805, 5.8091106, 4.7337327, 5.8150659, -0.7423279, 0.7446115

Time for backsubstitution: 22.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 6166

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of NS_A1_A1_A1_B1

### Relational analysis result of NS_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4433478, upper bound: 0.4421469
time: 3.30 seconds

## Relational analysis of NS_A1_A1_A1_B2

### Relational analysis result of NS_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4433477, upper bound: 0.4432076
time: 2.98 seconds

## BFS NS instance: NS_A1_A1_A2

### Backsubstitution after applying NS history:
0: -12.1845436, -9.9173183, -12.1932411, -9.9087467, -1.5491071, 1.5183749
1: -11.3617287, -9.5809584, -11.3665113, -9.5755863, -1.1759539, 1.1668282
2: -7.7876158, -6.4826922, -7.8013039, -6.4764924, -0.8738072, 0.9126740
3: -7.2851419, -5.5007691, -7.3087721, -5.4940162, -1.3781722, 1.3451662
4: -3.2377429, -1.6555676, -3.2478361, -1.6377206, -1.3810267, 1.3584762
5: -5.6457953, -4.1815443, -5.6508489, -4.1723309, -1.2505035, 1.2171772
6: -16.5179577, -14.3080711, -16.5311680, -14.2924919, -1.2308302, 1.2224896
7: -4.3251414, -2.6934607, -4.3496900, -2.6772752, -1.2953401, 1.3084023
8: -4.9502473, -3.3555560, -4.9596300, -3.3483191, -0.8993433, 0.8966769
9: 4.7376623, 5.8160009, 4.7312565, 5.8183122, -0.7540298, 0.7542897

Time for backsubstitution: 22.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 6166

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of NS_A1_A1_A2_B1

### Relational analysis result of NS_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441460, upper bound: 0.4421468
time: 5.13 seconds

## Relational analysis of NS_A1_A1_A2_B2

### Relational analysis result of NS_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441460, upper bound: 0.4449266
time: 3.74 seconds

## BFS NS instance: NS_A1_A2_A1

### Backsubstitution after applying NS history:
0: -12.1738539, -9.9486656, -12.1952343, -9.9298820, -1.5091910, 1.5251355
1: -11.3500919, -9.6016674, -11.3652792, -9.5863943, -1.1543317, 1.1561277
2: -7.7521563, -6.4915223, -7.7809014, -6.4776082, -0.8692396, 0.8842950
3: -7.2643046, -5.5496402, -7.3072338, -5.5221500, -1.3289480, 1.3471234
4: -3.2283404, -1.6599212, -3.2438068, -1.6402912, -1.3572478, 1.3507216
5: -5.6219597, -4.2187052, -5.6463623, -4.1920323, -1.2089376, 1.2103961
6: -16.5148563, -14.3109140, -16.5307655, -14.2979670, -1.2160223, 1.2249866
7: -4.3144064, -2.7109048, -4.3433723, -2.6815774, -1.2832007, 1.2829454
8: -4.9438796, -3.3578033, -4.9575539, -3.3510590, -0.8859236, 0.8910375
9: 4.7429008, 5.8104725, 4.7336011, 5.8155899, -0.7457325, 0.7469218

Time for backsubstitution: 22.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 6166

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of NS_A1_A2_A1_B1

### Relational analysis result of NS_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4433478, upper bound: 0.4424888
time: 3.07 seconds

## Relational analysis of NS_A1_A2_A1_B2

### Relational analysis result of NS_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4433477, upper bound: 0.4435450
time: 3.10 seconds

## BFS NS instance: NS_A1_A2_A2

### Backsubstitution after applying NS history:
0: -12.1966133, -9.9053307, -12.1980705, -9.9086866, -1.5555940, 1.5349936
1: -11.3642521, -9.5780764, -11.3671646, -9.5754204, -1.1799965, 1.1707785
2: -7.7941675, -6.4747620, -7.8041115, -6.4763632, -0.8776906, 0.9185083
3: -7.2938023, -5.4900236, -7.3122149, -5.4939137, -1.3832695, 1.3576174
4: -3.2408972, -1.6536186, -3.2480054, -1.6370277, -1.3849669, 1.3605120
5: -5.6495295, -4.1767702, -5.6521759, -4.1722374, -1.2565072, 1.2245710
6: -16.5265408, -14.2986269, -16.5345631, -14.2923393, -1.2355173, 1.2350817
7: -4.3276525, -2.6908753, -4.3500471, -2.6770909, -1.2973223, 1.3108563
8: -4.9537477, -3.3509407, -4.9597845, -3.3471813, -0.9041603, 0.9006755
9: 4.7350769, 5.8173742, 4.7311230, 5.8188343, -0.7574515, 0.7566108

Time for backsubstitution: 22.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 6166

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of NS_A1_A2_A2_B1

### Relational analysis result of NS_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441460, upper bound: 0.4441587
time: 3.87 seconds

## Relational analysis of NS_A1_A2_A2_B2

### Relational analysis result of NS_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441460, upper bound: 0.4452796
time: 4.28 seconds

## BFS NS instance: NS_A2_A1_A1

### Backsubstitution after applying NS history:
0: -12.1671162, -9.9474478, -12.1922560, -9.9249325, -1.5032539, 1.5080924
1: -11.3539600, -9.5981541, -11.3669291, -9.5842686, -1.1720643, 1.1691780
2: -7.7575202, -6.4892516, -7.7820296, -6.4735918, -0.8805852, 0.8876518
3: -7.2876787, -5.5524902, -7.3167372, -5.5209990, -1.3420324, 1.3534698
4: -3.2377808, -1.6338408, -3.2463644, -1.6294932, -1.3802490, 1.3726592
5: -5.6225452, -4.2073212, -5.6459751, -4.1850038, -1.2030973, 1.2080221
6: -16.5188732, -14.2942743, -16.5306950, -14.2874365, -1.2367959, 1.2289271
7: -4.3520365, -2.6928825, -4.3595705, -2.6772151, -1.2989678, 1.3125463
8: -4.9561658, -3.3555679, -4.9639397, -3.3504024, -0.8947802, 0.9067657
9: 4.7351770, 5.8108306, 4.7296653, 5.8155031, -0.7549458, 0.7513897

Time for backsubstitution: 22.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 5799

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 885

## Relational analysis of NS_A2_A1_A1_A1

### Relational analysis result of NS_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4435856, upper bound: 0.4435494
time: 3.37 seconds

## Relational analysis of NS_A2_A1_A1_A2

### Relational analysis result of NS_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4452141, upper bound: 0.4440523
time: 3.12 seconds

## BFS NS instance: NS_A2_A1_A2

### Backsubstitution after applying NS history:
0: -12.1898746, -9.9036427, -12.1950884, -9.9035597, -1.5509176, 1.5182719
1: -11.3681078, -9.5726833, -11.3688107, -9.5724983, -1.1982198, 1.1846812
2: -7.8040061, -6.4724908, -7.8070540, -6.4723463, -0.8917317, 0.9280035
3: -7.3185644, -5.4928846, -7.3222866, -5.4927664, -1.3976150, 1.3658912
4: -3.2503834, -1.6265454, -3.2505665, -1.6258054, -1.4083457, 1.3832359
5: -5.6505380, -4.1653070, -5.6519699, -4.1651978, -1.2532220, 1.2227249
6: -16.5310345, -14.2819843, -16.5347099, -14.2818127, -1.2571130, 1.2394359
7: -4.3662705, -2.6729598, -4.3666472, -2.6727548, -1.3141279, 1.3411584
8: -4.9659834, -3.3470616, -4.9661579, -3.3458490, -0.9138997, 0.9178249
9: 4.7273369, 5.8184881, 4.7271881, 5.8190517, -0.7670898, 0.7622056

Time for backsubstitution: 22.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 6166

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4628

## Relational analysis of NS_A2_A1_A2_B1

### Relational analysis result of NS_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4459312, upper bound: 0.4444247
time: 4.43 seconds

## Relational analysis of NS_A2_A1_A2_B2

### Relational analysis result of NS_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4459312, upper bound: 0.4455742
time: 4.87 seconds

## BFS NS instance: NS_A2_A2_A1

### Backsubstitution after applying NS history:
0: -12.1791935, -9.9354687, -12.1970816, -9.9248753, -1.5097322, 1.5247140
1: -11.3564796, -9.5952797, -11.3675823, -9.5841007, -1.1761203, 1.1731131
2: -7.7639656, -6.4813213, -7.7848411, -6.4734616, -0.8843920, 0.8982650
3: -7.2963243, -5.5417490, -7.3201766, -5.5208960, -1.3471265, 1.3675148
4: -3.2409263, -1.6318984, -3.2465329, -1.6287966, -1.3842106, 1.3746951
5: -5.6262913, -4.2025366, -5.6473022, -4.1849098, -1.2104235, 1.2154007
6: -16.5274601, -14.2848263, -16.5340919, -14.2872849, -1.2414703, 1.2415278
7: -4.3545341, -2.6902966, -4.3599267, -2.6770337, -1.3009377, 1.3153524
8: -4.9596524, -3.3509941, -4.9640884, -3.3492670, -0.8995938, 0.9107323
9: 4.7325959, 5.8121910, 4.7295332, 5.8160295, -0.7583477, 0.7537038

Time for backsubstitution: 22.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 5799

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 885

## Relational analysis of NS_A2_A2_A1_A1

### Relational analysis result of NS_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4435856, upper bound: 0.4438925
time: 4.22 seconds

## Relational analysis of NS_A2_A2_A1_A2

### Relational analysis result of NS_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4452141, upper bound: 0.4443967
time: 2.99 seconds

## BFS NS instance: NS_A2_A2_A2

### Backsubstitution after applying NS history:
0: -12.2019444, -9.8916492, -12.1999149, -9.9034986, -1.5574079, 1.5348949
1: -11.3706188, -9.5697899, -11.3694639, -9.5723333, -1.2022614, 1.1886454
2: -7.8105454, -6.4645619, -7.8098617, -6.4722161, -0.8955996, 0.9357066
3: -7.3272109, -5.4821386, -7.3257260, -5.4926634, -1.4027145, 1.3799405
4: -3.2535405, -1.6245813, -3.2507365, -1.6251118, -1.4123030, 1.3852680
5: -5.6542740, -4.1605124, -5.6532955, -4.1651039, -1.2590537, 1.2301230
6: -16.5396099, -14.2725353, -16.5381012, -14.2816563, -1.2617881, 1.2520330
7: -4.3687668, -2.6703370, -4.3670034, -2.6725693, -1.3161168, 1.3439913
8: -4.9694891, -3.3424611, -4.9663134, -3.3447127, -0.9187202, 0.9218471
9: 4.7247477, 5.8198586, 4.7270522, 5.8195767, -0.7705081, 0.7645316

Time for backsubstitution: 22.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 6166

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4628

## Relational analysis of NS_A2_A2_A2_B1

### Relational analysis result of NS_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4459312, upper bound: 0.4432431
time: 6.73 seconds

## Relational analysis of NS_A2_A2_A2_B2

### Relational analysis result of NS_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4459312, upper bound: 0.4459307
time: 4.57 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 33.75 seconds
NS_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 33.75
Output dim: 9, lower bound: -0.4433478, upper bound: 0.4421469
NS_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 33.75
Output dim: 9, lower bound: -0.4433477, upper bound: 0.4432076
NS_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.75
Output dim: 9, lower bound: -0.4441460, upper bound: 0.4421468
NS_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.75
Output dim: 9, lower bound: -0.4441460, upper bound: 0.4449266
NS_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 33.75
Output dim: 9, lower bound: -0.4433478, upper bound: 0.4424888
NS_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 33.75
Output dim: 9, lower bound: -0.4433477, upper bound: 0.4435450
NS_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.75
Output dim: 9, lower bound: -0.4441460, upper bound: 0.4441587
NS_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.75
Output dim: 9, lower bound: -0.4441460, upper bound: 0.4452796
NS_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 33.75
Output dim: 9, lower bound: -0.4435856, upper bound: 0.4435494
NS_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 33.75
Output dim: 9, lower bound: -0.4452141, upper bound: 0.4440523
NS_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.75
Output dim: 9, lower bound: -0.4459312, upper bound: 0.4444247
NS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.75
Output dim: 9, lower bound: -0.4459312, upper bound: 0.4455742
NS_A2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 33.75
Output dim: 9, lower bound: -0.4435856, upper bound: 0.4438925
NS_A2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 33.75
Output dim: 9, lower bound: -0.4452141, upper bound: 0.4443967
NS_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.75
Output dim: 9, lower bound: -0.4459312, upper bound: 0.4432431
NS_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.75
Output dim: 9, lower bound: -0.4459312, upper bound: 0.4459307

## BFS NS instance: NS_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -12.1617756, -9.9606438, -12.1869173, -9.9381723, -1.4952798, 1.5044365
1: -11.3475599, -9.6045341, -11.3605385, -9.5905867, -1.1463566, 1.1476758
2: -7.7457032, -6.4994516, -7.7701368, -6.4837952, -0.8589447, 0.8664968
3: -7.2556467, -5.5603809, -7.2847137, -5.5288959, -1.3175821, 1.3153303
4: -3.2251983, -1.6618605, -3.2337437, -1.6575041, -1.3365798, 1.3399906
5: -5.6182165, -4.2234697, -5.6416759, -4.2012191, -1.1954679, 1.1975045
6: -16.5062695, -14.3203602, -16.5181503, -14.3135281, -1.1962013, 1.2032626
7: -4.3118916, -2.7134852, -4.3194327, -2.6977754, -1.2674832, 1.2571855
8: -4.9403973, -3.3623924, -4.9482164, -3.3572197, -0.8754427, 0.8805602
9: 4.7454805, 5.8091106, 4.7399850, 5.8137627, -0.7388492, 0.7394629

Time for backsubstitution: 21.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 6166

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 885

## Relational analysis of NS_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 885

### Candidate
type: A, layer: 1, pos: 4628

## Relational analysis of NS_A1_A1_A1_B1_A1

### Relational analysis result of NS_A1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4421957, upper bound: 0.4421451
time: 3.14 seconds

## Relational analysis of NS_A1_A1_A1_B1_A2

### Relational analysis result of NS_A1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4433455, upper bound: 0.4421447
time: 3.37 seconds

## BFS NS instance: NS_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -12.1617756, -9.9606438, -12.1919003, -9.9249420, -1.5077586, 1.5089076
1: -11.3475599, -9.6045341, -11.3665466, -9.5842686, -1.1525126, 1.1544087
2: -7.7457032, -6.4994516, -7.7815318, -6.4735942, -0.8698158, 0.8762412
3: -7.2556467, -5.5603809, -7.3166990, -5.5213137, -1.3245440, 1.3434963
4: -3.2251983, -1.6618605, -3.2461741, -1.6295123, -1.3648815, 1.3503604
5: -5.6182165, -4.2234697, -5.6459727, -4.1859140, -1.2061372, 1.2022552
6: -16.5062695, -14.3203602, -16.5302181, -14.2874527, -1.2156515, 1.2150922
7: -4.3118916, -2.7134852, -4.3594785, -2.6772339, -1.2843227, 1.2853663
8: -4.9403973, -3.3623924, -4.9635124, -3.3504233, -0.8819373, 0.8936217
9: 4.7454805, 5.8091106, 4.7297010, 5.8154888, -0.7417736, 0.7497497

Time for backsubstitution: 21.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 6166

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 885

## Relational analysis of NS_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 885

### Candidate
type: A, layer: 1, pos: 4628

## Relational analysis of NS_A1_A1_A1_B2_A1

### Relational analysis result of NS_A1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4421957, upper bound: 0.4432035
time: 3.28 seconds

## Relational analysis of NS_A1_A1_A1_B2_A2

### Relational analysis result of NS_A1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4433454, upper bound: 0.4432054
time: 3.22 seconds

## BFS NS instance: NS_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -12.1845436, -9.9173183, -12.1897554, -9.9172373, -1.5426435, 1.5142984
1: -11.3617287, -9.5809584, -11.3624363, -9.5807724, -1.1713076, 1.1623399
2: -7.7876158, -6.4826922, -7.7906618, -6.4825497, -0.8673198, 0.9035268
3: -7.2851419, -5.5007691, -7.2888680, -5.5006518, -1.3718989, 1.3262727
4: -3.2377429, -1.6555676, -3.2379246, -1.6548257, -1.3637390, 1.3497994
5: -5.6457953, -4.1815443, -5.6472254, -4.1814404, -1.2446070, 1.2110040
6: -16.5179577, -14.3080711, -16.5216293, -14.3079033, -1.2156959, 1.2125297
7: -4.3251414, -2.6934607, -4.3255167, -2.6932611, -1.2816172, 1.2848034
8: -4.9502473, -3.3555560, -4.9504180, -3.3543420, -0.8925641, 0.8901055
9: 4.7376623, 5.8160009, 4.7375159, 5.8165655, -0.7498732, 0.7491536

Time for backsubstitution: 21.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 6166

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 4628

## Relational analysis of NS_A1_A1_A2_B1_B1

### Relational analysis result of NS_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441438, upper bound: 0.4426531
time: 4.11 seconds

## Relational analysis of NS_A1_A1_A2_B1_B2

### Relational analysis result of NS_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441438, upper bound: 0.4421445
time: 5.08 seconds

## BFS NS instance: NS_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -12.1845436, -9.9173183, -12.1947355, -9.9035673, -1.5510736, 1.5187588
1: -11.3617287, -9.5809584, -11.3684292, -9.5725002, -1.1787329, 1.1689796
2: -7.7876158, -6.4826922, -7.8065677, -6.4723496, -0.8781784, 0.9161530
3: -7.2851419, -5.5007691, -7.3222456, -5.4930811, -1.3744173, 1.3545151
4: -3.2377429, -1.6555676, -3.2503786, -1.6258256, -1.3929877, 1.3601449
5: -5.6457953, -4.1815443, -5.6519651, -4.1661081, -1.2537038, 1.2168715
6: -16.5179577, -14.3080711, -16.5342331, -14.2818260, -1.2354581, 1.2257757
7: -4.3251414, -2.6934607, -4.3665557, -2.6727722, -1.2984037, 1.3101671
8: -4.9502473, -3.3555560, -4.9657311, -3.3458700, -0.9009335, 0.9032032
9: 4.7376623, 5.8160009, 4.7272205, 5.8190398, -0.7539239, 0.7594521

Time for backsubstitution: 21.73 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.41 + 547.54 = 604.95 seconds
