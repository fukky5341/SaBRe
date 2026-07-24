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
execution time: IAR + RelationalAnalysis = 22.58 + 33.68 = 56.26 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.4459361, upper bound: 0.4459358

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4628
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 4628

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4447841, upper bound: 0.4459337
time: 5.04 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4459338, upper bound: 0.4459335
time: 3.99 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.10 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 9.10
Output dim: 9, lower bound: -0.4447841, upper bound: 0.4459337
NS_A2, status: Status.UNKNOWN, split count: 1, time: 9.10
Output dim: 9, lower bound: -0.4459338, upper bound: 0.4459335

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -12.1954603, -9.9181957, -12.1994591, -9.9083748, -1.5517750, 1.5482464
1: -11.3671331, -9.5764179, -11.3689384, -9.5736427, -1.1928720, 1.1915808
2: -7.8066664, -6.4728904, -7.8089476, -6.4724321, -0.9270518, 0.9289073
3: -7.3099179, -5.4959412, -7.3205600, -5.4930382, -1.4047260, 1.4122005
4: -3.2474382, -1.6268952, -3.2498505, -1.6255145, -1.3976650, 1.3985019
5: -5.6468897, -4.1669493, -5.6513081, -4.1654010, -1.2555804, 1.2577496
6: -16.5353870, -14.2893248, -16.5376835, -14.2842045, -1.2574296, 1.2574761
7: -4.3642621, -2.6815789, -4.3665395, -2.6754363, -1.3363132, 1.3323195
8: -4.9632349, -3.3607802, -4.9657936, -3.3499928, -0.9133551, 0.9047095
9: 4.7345309, 5.8173580, 4.7294455, 5.8191948, -0.7579458, 0.7608058

Time for backsubstitution: 20.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441302, upper bound: 0.4441480
time: 3.24 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4447838, upper bound: 0.4459332
time: 3.34 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -12.1999378, -9.9034948, -12.1999397, -9.9034901, -1.5632615, 1.5558963
1: -11.3694658, -9.5723286, -11.3694668, -9.5723295, -1.1966591, 1.1948822
2: -7.8098764, -6.4722166, -7.8098774, -6.4722166, -0.9311519, 0.9355543
3: -7.3257322, -5.4926600, -7.3257384, -5.4926596, -1.4083428, 1.4207764
4: -3.2507372, -1.6251094, -3.2507381, -1.6251090, -1.4025960, 1.4012961
5: -5.6533003, -4.1651011, -5.6533022, -4.1650996, -1.2628264, 1.2652507
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2647538, 1.2657709
7: -4.3670063, -2.6725688, -4.3670073, -2.6725664, -1.3419933, 1.3387311
8: -4.9663134, -3.3447137, -4.9663148, -3.3447104, -0.9227240, 0.9156601
9: 4.7270527, 5.8195786, 4.7270503, 5.8195791, -0.7604957, 0.7659513

Time for backsubstitution: 22.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4628

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4459338, upper bound: 0.4447842
time: 4.23 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4459338, upper bound: 0.4459337
time: 4.26 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 30.56 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 30.56
Output dim: 9, lower bound: -0.4441302, upper bound: 0.4441480
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 30.56
Output dim: 9, lower bound: -0.4447838, upper bound: 0.4459332
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 30.56
Output dim: 9, lower bound: -0.4459338, upper bound: 0.4447842
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 30.56
Output dim: 9, lower bound: -0.4459338, upper bound: 0.4459337

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -12.1936159, -9.9233904, -12.1941252, -9.9220667, -1.5473027, 1.5473709
1: -11.3648415, -9.5795250, -11.3625698, -9.5819311, -1.1747284, 1.1736004
2: -7.8009267, -6.4770374, -7.7925663, -6.4826345, -0.9117262, 0.9108684
3: -7.2963896, -5.4971943, -7.2871342, -5.5009241, -1.3839903, 1.3788428
4: -3.2447236, -1.6388075, -3.2372267, -1.6545241, -1.3619089, 1.3712606
5: -5.6457691, -4.1740737, -5.6465635, -4.1816435, -1.2502518, 1.2522423
6: -16.5318584, -14.3000088, -16.5246181, -14.3102970, -1.2257395, 1.2311056
7: -4.3473077, -2.6861117, -4.3254147, -2.6959484, -1.3035550, 1.2896831
8: -4.9567070, -3.3632526, -4.9500499, -3.3584871, -0.8907869, 0.8821597
9: 4.7386036, 5.8166275, 4.7397757, 5.8167162, -0.7485056, 0.7503928

Time for backsubstitution: 21.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 885

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441290, upper bound: 0.4437910
time: 3.16 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441290, upper bound: 0.4441444
time: 3.59 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -12.1954603, -9.9181976, -12.1994534, -9.9083796, -1.5473394, 1.5480103
1: -11.3671312, -9.5764198, -11.3689375, -9.5736437, -1.1928229, 1.1958914
2: -7.8066664, -6.4728909, -7.8089471, -6.4724331, -0.9270439, 0.9294763
3: -7.3099174, -5.4959412, -7.3205562, -5.4930363, -1.4047265, 1.3985038
4: -3.2474384, -1.6268950, -3.2498484, -1.6255167, -1.3867006, 1.3985007
5: -5.6468892, -4.1669493, -5.6513071, -4.1654034, -1.2557063, 1.2550182
6: -16.5353870, -14.2893257, -16.5376816, -14.2842073, -1.2425561, 1.2574751
7: -4.3642607, -2.6815798, -4.3665338, -2.6754382, -1.3363132, 1.3084698
8: -4.9632359, -3.3607802, -4.9657912, -3.3499942, -0.9120443, 0.8966029
9: 4.7345319, 5.8173580, 4.7294464, 5.8191948, -0.7564189, 0.7634692

Time for backsubstitution: 21.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4447826, upper bound: 0.4455778
time: 3.04 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4447826, upper bound: 0.4459320
time: 3.22 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -12.1999378, -9.9034948, -12.1954603, -9.9181957, -1.5483727, 1.5587287
1: -11.3694658, -9.5723286, -11.3671331, -9.5764179, -1.1921215, 1.1943500
2: -7.8098764, -6.4722166, -7.8066664, -6.4728904, -0.9301670, 0.9269032
3: -7.3257322, -5.4926600, -7.3099179, -5.4959412, -1.4173307, 1.4051247
4: -3.2507372, -1.6251094, -3.2474382, -1.6268952, -1.3991137, 1.3980424
5: -5.6533003, -4.1651011, -5.6468897, -4.1669493, -1.2606621, 1.2556176
6: -16.5381165, -14.2816582, -16.5353870, -14.2893248, -1.2572765, 1.2618573
7: -4.3670063, -2.6725688, -4.3642621, -2.6815789, -1.3328080, 1.3392472
8: -4.9663134, -3.3447137, -4.9632349, -3.3607802, -0.9054186, 0.9190779
9: 4.7270527, 5.8195786, 4.7345309, 5.8173580, -0.7633777, 0.7584004

Time for backsubstitution: 21.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441457, upper bound: 0.4441303
time: 3.19 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4459331, upper bound: 0.4447838
time: 3.24 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -12.1999378, -9.9034948, -12.1999378, -9.9034948, -1.5558944, 1.5558941
1: -11.3694658, -9.5723286, -11.3694658, -9.5723286, -1.1948819, 1.1948822
2: -7.8098764, -6.4722166, -7.8098764, -6.4722166, -0.9355524, 0.9355524
3: -7.3257322, -5.4926600, -7.3257322, -5.4926600, -1.4083433, 1.4083433
4: -3.2507372, -1.6251094, -3.2507372, -1.6251094, -1.4025955, 1.4025955
5: -5.6533003, -4.1651011, -5.6533003, -4.1651011, -1.2652473, 1.2652471
6: -16.5381165, -14.2816582, -16.5381165, -14.2816582, -1.2657697, 1.2657697
7: -4.3670063, -2.6725688, -4.3670063, -2.6725688, -1.3387308, 1.3387306
8: -4.9663134, -3.3447137, -4.9663134, -3.3447137, -0.9156594, 0.9156594
9: 4.7270527, 5.8195786, 4.7270527, 5.8195786, -0.7604949, 0.7604949

Time for backsubstitution: 22.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441459, upper bound: 0.4441304
time: 3.23 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4459333, upper bound: 0.4447853
time: 3.17 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 28.61 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.61
Output dim: 9, lower bound: -0.4441290, upper bound: 0.4437910
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.61
Output dim: 9, lower bound: -0.4441290, upper bound: 0.4441444
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.61
Output dim: 9, lower bound: -0.4447826, upper bound: 0.4455778
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.61
Output dim: 9, lower bound: -0.4447826, upper bound: 0.4459320
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 28.61
Output dim: 9, lower bound: -0.4441457, upper bound: 0.4441303
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 28.61
Output dim: 9, lower bound: -0.4459331, upper bound: 0.4447838
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.61
Output dim: 9, lower bound: -0.4441459, upper bound: 0.4441304
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.61
Output dim: 9, lower bound: -0.4459333, upper bound: 0.4447853

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -12.1835537, -9.9235134, -12.1892738, -9.9221220, -1.5376134, 1.5426250
1: -11.3634834, -9.5798664, -11.3619146, -9.5820951, -1.1719828, 1.1714749
2: -7.7950830, -6.4773102, -7.7897515, -6.4827652, -0.9055560, 0.9077260
3: -7.2892208, -5.4974089, -7.2836771, -5.5010262, -1.3764567, 1.3749022
4: -3.2443790, -1.6402402, -3.2370572, -1.6552234, -1.3606663, 1.3694754
5: -5.6430087, -4.1742687, -5.6452332, -4.1817346, -1.2434740, 1.2470026
6: -16.5247784, -14.3003244, -16.5212059, -14.3104496, -1.2186224, 1.2275803
7: -4.3465796, -2.6864953, -4.3250527, -2.6961315, -1.3025408, 1.2888556
8: -4.9563770, -3.3656015, -4.9498930, -3.3596287, -0.8891456, 0.8794397
9: 4.7388849, 5.8155375, 4.7399111, 5.8161912, -0.7468293, 0.7481961

Time for backsubstitution: 21.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 885

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4628

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441290, upper bound: 0.4426437
time: 3.30 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441290, upper bound: 0.4437910
time: 3.25 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -12.1956224, -9.9115200, -12.1941032, -9.9220657, -1.5440707, 1.5592654
1: -11.3659954, -9.5769711, -11.3625698, -9.5819321, -1.1760254, 1.1754062
2: -7.8016152, -6.4693813, -7.7925563, -6.4826360, -0.9093986, 0.9183389
3: -7.2978821, -5.4866610, -7.2871237, -5.5009241, -1.3815675, 1.3889585
4: -3.2475307, -1.6382790, -3.2372265, -1.6545267, -1.3646059, 1.3715117
5: -5.6467471, -4.1694880, -5.6465597, -4.1816430, -1.2519217, 1.2544210
6: -16.5333538, -14.2908764, -16.5246048, -14.3102970, -1.2233133, 1.2377149
7: -4.3490777, -2.6838975, -4.3254132, -2.6959476, -1.3045158, 1.2916937
8: -4.9598842, -3.3609939, -4.9500499, -3.3584905, -0.8939631, 0.8834530
9: 4.7362909, 5.8169088, 4.7397761, 5.8167133, -0.7502539, 0.7505212

Time for backsubstitution: 21.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 885

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4628

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441290, upper bound: 0.4429951
time: 3.82 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441290, upper bound: 0.4441444
time: 3.63 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -12.1854029, -9.9183178, -12.1946068, -9.9084358, -1.5376477, 1.5432615
1: -11.3657742, -9.5767584, -11.3682833, -9.5738087, -1.1900806, 1.1937644
2: -7.8008213, -6.4731641, -7.8061295, -6.4725628, -0.9208724, 0.9263465
3: -7.3027544, -5.4961572, -7.3171062, -5.4931402, -1.3971972, 1.3945696
4: -3.2470918, -1.6283281, -3.2496789, -1.6262124, -1.3854613, 1.3967180
5: -5.6441283, -4.1671476, -5.6499743, -4.1654987, -1.2489076, 1.2497694
6: -16.5283051, -14.2896404, -16.5342751, -14.2843599, -1.2354364, 1.2539465
7: -4.3635364, -2.6819606, -4.3661761, -2.6756244, -1.3352966, 1.3076422
8: -4.9629054, -3.3631287, -4.9656329, -3.3511329, -0.9103954, 0.8938718
9: 4.7348137, 5.8162689, 4.7295818, 5.8186703, -0.7547388, 0.7612768

Time for backsubstitution: 21.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 885

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4628

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4447826, upper bound: 0.4444281
time: 3.02 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4447826, upper bound: 0.4455778
time: 3.08 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -12.1974659, -9.9063206, -12.1994343, -9.9083786, -1.5441089, 1.5599000
1: -11.3682842, -9.5738611, -11.3689346, -9.5736446, -1.1941166, 1.1977079
2: -7.8073492, -6.4652357, -7.8089352, -6.4724345, -0.9247119, 0.9369497
3: -7.3114100, -5.4854107, -7.3205462, -5.4930391, -1.4023066, 1.4086199
4: -3.2502451, -1.6263654, -3.2498488, -1.6255178, -1.3894148, 1.3987520
5: -5.6478682, -4.1623578, -5.6513033, -4.1654029, -1.2573395, 1.2571888
6: -16.5368805, -14.2801933, -16.5376720, -14.2842093, -1.2401152, 1.2665515
7: -4.3660288, -2.6793184, -4.3665318, -2.6754379, -1.3372755, 1.3104823
8: -4.9664135, -3.3585258, -4.9657907, -3.3499975, -0.9152203, 0.8979255
9: 4.7322173, 5.8176384, 4.7294469, 5.8191953, -0.7581708, 0.7636007

Time for backsubstitution: 23.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 885

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4628

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4447826, upper bound: 0.4447828
time: 3.30 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4447826, upper bound: 0.4459339
time: 3.24 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -12.1946030, -9.9171753, -12.1936159, -9.9233904, -1.5474949, 1.5542583
1: -11.3630905, -9.5806036, -11.3648415, -9.5795250, -1.1741409, 1.1762230
2: -7.7934828, -6.4824190, -7.8009267, -6.4770374, -0.9121947, 0.9115740
3: -7.2923222, -5.5005445, -7.2963896, -5.4971943, -1.3839846, 1.3843904
4: -3.2380962, -1.6541243, -3.2447236, -1.6388075, -1.3718591, 1.3622830
5: -5.6485581, -4.1813464, -5.6457691, -4.1740737, -1.2551489, 1.2502794
6: -16.5250397, -14.3077507, -16.5318584, -14.3000088, -1.2308979, 1.2301610
7: -4.3258801, -2.6930799, -4.3473077, -2.6861117, -1.2901678, 1.3064914
8: -4.9505754, -3.3532023, -4.9567070, -3.3632526, -0.8828895, 0.8965138
9: 4.7373800, 5.8170900, 4.7386036, 5.8166275, -0.7529619, 0.7489569

Time for backsubstitution: 23.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 885

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4437907, upper bound: 0.4441292
time: 3.29 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441445, upper bound: 0.4441312
time: 3.35 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -12.1999388, -9.9034939, -12.1954603, -9.9181976, -1.5481372, 1.5542936
1: -11.3694658, -9.5723305, -11.3671312, -9.5764198, -1.1964321, 1.1943009
2: -7.8098755, -6.4722176, -7.8066664, -6.4728909, -0.9307363, 0.9268954
3: -7.3257284, -5.4926610, -7.3099174, -5.4959412, -1.4036341, 1.4051251
4: -3.2507367, -1.6251111, -3.2474384, -1.6268950, -1.3991141, 1.3870790
5: -5.6533012, -4.1651039, -5.6468892, -4.1669493, -1.2579312, 1.2557497
6: -16.5381165, -14.2816620, -16.5353870, -14.2893257, -1.2572756, 1.2469833
7: -4.3670015, -2.6725702, -4.3642607, -2.6815798, -1.3089590, 1.3392472
8: -4.9663115, -3.3447132, -4.9632359, -3.3607802, -0.8973415, 0.9177672
9: 4.7270536, 5.8195782, 4.7345319, 5.8173580, -0.7660403, 0.7568762

Time for backsubstitution: 23.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4455756, upper bound: 0.4447847
time: 3.24 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4459319, upper bound: 0.4447838
time: 3.16 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -12.1946030, -9.9171753, -12.1980915, -9.9086771, -1.5550232, 1.5513840
1: -11.3630905, -9.5806036, -11.3671665, -9.5754204, -1.1769190, 1.1767671
2: -7.7934828, -6.4824190, -7.8041286, -6.4763622, -0.9175704, 0.9202163
3: -7.2923222, -5.5005445, -7.3122225, -5.4939103, -1.3749981, 1.3876209
4: -3.2380962, -1.6541243, -3.2480063, -1.6370230, -1.3753452, 1.3668287
5: -5.6485581, -4.1813464, -5.6521816, -4.1722336, -1.2597141, 1.2599053
6: -16.5250397, -14.3077507, -16.5345764, -14.2923412, -1.2394080, 1.2340691
7: -4.3258801, -2.6930799, -4.3500500, -2.6770911, -1.2960963, 1.3059685
8: -4.9505754, -3.3532023, -4.9597859, -3.3471813, -0.8931329, 0.8930837
9: 4.7373800, 5.8170900, 4.7311234, 5.8188391, -0.7500594, 0.7510463

Time for backsubstitution: 23.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4437909, upper bound: 0.4441307
time: 3.05 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441447, upper bound: 0.4441309
time: 3.23 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -12.1999388, -9.9034939, -12.1999397, -9.9034910, -1.5556583, 1.5514598
1: -11.3694658, -9.5723305, -11.3694649, -9.5723295, -1.1991930, 1.1948333
2: -7.8098755, -6.4722176, -7.8098764, -6.4722152, -0.9361215, 0.9355443
3: -7.3257284, -5.4926610, -7.3257341, -5.4926610, -1.3946466, 1.4083424
4: -3.2507367, -1.6251111, -3.2507381, -1.6251090, -1.4025950, 1.3916316
5: -5.6533012, -4.1651039, -5.6533012, -4.1651011, -1.2625170, 1.2653828
6: -16.5381165, -14.2816620, -16.5381165, -14.2816572, -1.2657681, 1.2508960
7: -4.3670015, -2.6725702, -4.3670068, -2.6725693, -1.3148823, 1.3387294
8: -4.9663115, -3.3447132, -4.9663153, -3.3447137, -0.9075847, 0.9143490
9: 4.7270536, 5.8195782, 4.7270522, 5.8195791, -0.7631299, 0.7589710

Time for backsubstitution: 23.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4455757, upper bound: 0.4447847
time: 3.43 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4459321, upper bound: 0.4447844
time: 3.04 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 29.61 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 9, lower bound: -0.4441290, upper bound: 0.4426437
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 9, lower bound: -0.4441290, upper bound: 0.4437910
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 9, lower bound: -0.4441290, upper bound: 0.4429951
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 9, lower bound: -0.4441290, upper bound: 0.4441444
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 9, lower bound: -0.4447826, upper bound: 0.4444281
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 9, lower bound: -0.4447826, upper bound: 0.4455778
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 9, lower bound: -0.4447826, upper bound: 0.4447828
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 9, lower bound: -0.4447826, upper bound: 0.4459339
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 9, lower bound: -0.4437907, upper bound: 0.4441292
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 9, lower bound: -0.4441445, upper bound: 0.4441312
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 9, lower bound: -0.4455756, upper bound: 0.4447847
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 9, lower bound: -0.4459319, upper bound: 0.4447838
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 9, lower bound: -0.4437909, upper bound: 0.4441307
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 9, lower bound: -0.4441447, upper bound: 0.4441309
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 9, lower bound: -0.4455757, upper bound: 0.4447847
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.61
Output dim: 9, lower bound: -0.4459321, upper bound: 0.4447844

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -12.1835537, -9.9235134, -12.1852837, -9.9319611, -1.5296502, 1.5382285
1: -11.3634834, -9.5798664, -11.3601217, -9.5848961, -1.1688929, 1.1697090
2: -7.7950830, -6.4773102, -7.7874937, -6.4832253, -0.9047377, 0.9050341
3: -7.2892208, -5.4974089, -7.2730069, -5.5039339, -1.3734078, 1.3643613
4: -3.2443790, -1.6402402, -3.2346826, -1.6565883, -1.3590670, 1.3670702
5: -5.6430087, -4.1742687, -5.6408119, -4.1832681, -1.2416415, 1.2429934
6: -16.5247784, -14.3003244, -16.5189247, -14.3155727, -1.2155809, 1.2244918
7: -4.3465796, -2.6864953, -4.3227797, -2.7022872, -1.2962794, 1.2866111
8: -4.9563770, -3.3656015, -4.9473238, -3.3704243, -0.8775642, 0.8764845
9: 4.7388849, 5.8155375, 4.7450008, 5.8143711, -0.7447190, 0.7432114

Time for backsubstitution: 22.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 885

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6166

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4432531, upper bound: 0.4409839
time: 4.65 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441280, upper bound: 0.4426426
time: 3.29 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -12.1835537, -9.9235134, -12.1897545, -9.9172325, -1.5445676, 1.5427477
1: -11.3634834, -9.5798664, -11.3624363, -9.5807695, -1.1734767, 1.1720147
2: -7.7950830, -6.4773102, -7.7906637, -6.4825497, -0.9054036, 0.9090490
3: -7.2892208, -5.4974089, -7.2888656, -5.5006485, -1.3768568, 1.3800440
4: -3.2443790, -1.6402402, -3.2379262, -1.6548221, -1.3610411, 1.3700738
5: -5.6430087, -4.1742687, -5.6472268, -4.1814394, -1.2435017, 1.2499084
6: -16.5247784, -14.3003244, -16.5216331, -14.3078995, -1.2230446, 1.2273715
7: -4.3465796, -2.6864953, -4.3255191, -2.6932631, -1.3054767, 1.2893400
8: -4.9563770, -3.3656015, -4.9504189, -3.3543444, -0.8948729, 0.8801676
9: 4.7388849, 5.8155375, 4.7375154, 5.8165665, -0.7472808, 0.7507657

Time for backsubstitution: 21.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 885

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 6166

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4432531, upper bound: 0.4421332
time: 3.70 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441280, upper bound: 0.4437919
time: 3.66 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -12.1956224, -9.9115200, -12.1901131, -9.9319000, -1.5361090, 1.5548687
1: -11.3659954, -9.5769711, -11.3607759, -9.5847301, -1.1729336, 1.1736398
2: -7.8016152, -6.4693813, -7.7902956, -6.4830947, -0.9085803, 0.9156396
3: -7.2978821, -5.4866610, -7.2764521, -5.5038328, -1.3785176, 1.3784173
4: -3.2475307, -1.6382790, -3.2348509, -1.6558900, -1.3630066, 1.3691053
5: -5.6467471, -4.1694880, -5.6421404, -4.1831756, -1.2500887, 1.2504106
6: -16.5333538, -14.2908764, -16.5223236, -14.3154202, -1.2202709, 1.2346013
7: -4.3490777, -2.6838975, -4.3231406, -2.7021041, -1.2982554, 1.2894497
8: -4.9598842, -3.3609939, -4.9474812, -3.3692870, -0.8823819, 0.8805007
9: 4.7362909, 5.8169088, 4.7448673, 5.8148947, -0.7481439, 0.7455354

Time for backsubstitution: 22.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 885

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6166

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4432531, upper bound: 0.4413253
time: 3.09 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441280, upper bound: 0.4429943
time: 3.52 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -12.1956224, -9.9115200, -12.1945820, -9.9171715, -1.5510273, 1.5588112
1: -11.3659954, -9.5769711, -11.3630915, -9.5806065, -1.1775198, 1.1759460
2: -7.8016152, -6.4693813, -7.7934713, -6.4824181, -0.9092467, 0.9196649
3: -7.2978821, -5.4866610, -7.2923117, -5.5005465, -1.3819666, 1.3916037
4: -3.2475307, -1.6382790, -3.2380962, -1.6541264, -1.3649807, 1.3721106
5: -5.6467471, -4.1694880, -5.6485538, -4.1813464, -1.2519493, 1.2573264
6: -16.5333538, -14.2908764, -16.5250244, -14.3077507, -1.2277353, 1.2365749
7: -4.3490777, -2.6838975, -4.3258796, -2.6930804, -1.3074503, 1.2921779
8: -4.9598842, -3.3609939, -4.9505768, -3.3532047, -0.8996904, 0.8841830
9: 4.7362909, 5.8169088, 4.7373810, 5.8170891, -0.7507055, 0.7530904

Time for backsubstitution: 21.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 885

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 6166

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4432531, upper bound: 0.4424751
time: 3.36 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.4441280, upper bound: 0.4441457
time: 3.07 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -12.1854029, -9.9183178, -12.1906118, -9.9182568, -1.5297174, 1.5388594
1: -11.3657742, -9.5767584, -11.3664761, -9.5765839, -1.1870227, 1.1919973
2: -7.8008213, -6.4731641, -7.8038530, -6.4730225, -0.9200473, 0.9236658
3: -7.3027544, -5.4961572, -7.3064632, -5.4960451, -1.3941507, 1.3840492
4: -3.2470918, -1.6283281, -3.2472699, -1.6275916, -1.3838592, 1.3942788
5: -5.6441283, -4.1671476, -5.6455584, -4.1670465, -1.2470527, 1.2457490
6: -16.5283051, -14.2896404, -16.5319786, -14.2894764, -1.2323918, 1.2508547
7: -4.3635364, -2.6819606, -4.3638983, -2.6817641, -1.3290462, 1.3053858
8: -4.9629054, -3.3631287, -4.9630752, -3.3619199, -0.8988197, 0.8908834
9: 4.7348137, 5.8162689, 4.7346678, 5.8168335, -0.7526176, 0.7563022

Time for backsubstitution: 22.25 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.26 + 550.48 = 606.75 seconds
