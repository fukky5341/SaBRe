## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.8650754865
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.6260853, -9.4460907, -12.6260853, -9.4460907, -3.1799946, 3.1799946)
1: (-11.7355261, -9.1776123, -11.7355261, -9.1776123, -2.4480700, 2.4480703)
2: (-8.1624212, -6.1997881, -8.1624212, -6.1997881, -1.9626331, 1.9626331)
3: (-7.7233071, -5.1149702, -7.7233071, -5.1149702, -2.6083369, 2.6083369)
4: (-3.6771948, -1.3426085, -3.6771948, -1.3426085, -2.3345864, 2.3345864)
5: (-5.9543295, -3.8286073, -5.9543295, -3.8286073, -2.1257222, 2.1257222)
6: (-16.9029446, -13.7977066, -16.9029446, -13.7977066, -3.0488920, 3.0488920)
7: (-4.6868649, -2.2577319, -4.6868649, -2.2577319, -2.4291329, 2.4291329)
8: (-5.2317653, -2.9253664, -5.2317653, -2.9253664, -2.1087542, 2.1087539)
9: (4.4055529, 5.9714375, 4.4055529, 5.9714375, -1.5658846, 1.5658846)

## BASE Result
execution time: IAR + LP analysis = 13.70 + 33.49 = 47.20 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -1.2026377, upper bound: 1.2026362


# Binary Search by BASE starts (time budget: 3552.80 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=1.3057825565338135
rel_dist={9: [-0.8660068367423657, 0.8660078437084611]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.VERIFIED, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.1616536378860474
rel_dist={9: [-0.6705902224208895, 0.6705927823695186]}

## Binary search (step 2) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=4, k_high=5, k_mid=4, eps_mid=0.0156250, abs_max=1.2096967697143555
rel_dist={9: [-0.7384599414700519, 0.738462472703862]}

## Binary search (step 3) starts
Candidate k: 5, corresponding eps: 0.0195312


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=5, k_high=5, k_mid=5, eps_mid=0.0195312, abs_max=1.2577396631240845
rel_dist={9: [-0.803270111495487, 0.8032729720962895]}

## Binary Search Result
Binary search time: 204.43 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01953125


# Individual Split (IS_dual) starts
Time budget: 3348.37 seconds

## Binary search (step 0) starts
Candidate k: 9, corresponding eps: 0.0351562


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6166
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 5799
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 6166

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0401546, upper bound: 1.0378345
time: 5.01 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0414148, upper bound: 1.0414154
time: 4.48 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 9.69 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 9.69
Output dim: 9, lower bound: -1.0401546, upper bound: 1.0378345
IS_A2, status: Status.UNKNOWN, split count: 1, time: 9.69
Output dim: 9, lower bound: -1.0414148, upper bound: 1.0414154

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -12.5779991, -9.5066519, -12.6220846, -9.4579229, -2.7326035, 2.8936734
1: -11.7001781, -9.2250204, -11.7314816, -9.1848726, -2.2128935, 2.1597073
2: -8.0719109, -6.2380009, -8.1453552, -6.2022729, -1.8371921, 1.8929951
3: -7.6701622, -5.2049522, -7.7196903, -5.1315050, -2.3761597, 2.3116958
4: -3.6570044, -1.3592873, -3.6744514, -1.3455408, -2.3114636, 2.3151641
5: -5.9032869, -3.8842707, -5.9499168, -3.8396583, -2.0636287, 2.0656462
6: -16.8792248, -13.8215885, -16.8996525, -13.8021393, -2.6038752, 2.5279360
7: -4.6579971, -2.2996969, -4.6814833, -2.2616923, -2.3963048, 2.3268585
8: -5.2140493, -2.9587946, -5.2304354, -2.9316587, -1.8337045, 1.8331325
9: 4.4324579, 5.9562163, 4.4102173, 5.9689026, -1.4214811, 1.4273851

Time for backsubstitution: 14.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5799
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0388888, upper bound: 1.0334002
time: 4.40 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0401535, upper bound: 1.0378338
time: 4.94 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -12.6260767, -9.4461374, -12.6260853, -9.4460907, -2.9584785, 2.9360204
1: -11.7355165, -9.1776304, -11.7355261, -9.1776123, -2.2070432, 2.2021430
2: -8.1623936, -6.1997938, -8.1624212, -6.1997881, -1.9111052, 1.9482012
3: -7.7232981, -5.1149902, -7.7233071, -5.1149702, -2.4407845, 2.3985786
4: -3.6771874, -1.3426166, -3.6771948, -1.3426085, -2.3345790, 2.3345783
5: -5.9543214, -3.8286192, -5.9543295, -3.8286073, -2.1257141, 2.1257102
6: -16.9029369, -13.7977180, -16.9029446, -13.7977066, -2.6418839, 2.6677327
7: -4.6868491, -2.2577415, -4.6868649, -2.2577319, -2.4291172, 2.4291234
8: -5.2317619, -2.9253788, -5.2317653, -2.9253664, -1.8601370, 1.8587091
9: 4.4055643, 5.9714341, 4.4055529, 5.9714375, -1.4502215, 1.4499041

Time for backsubstitution: 13.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6166
type: A, layer: 1, pos: 5799
type: B, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5860

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6166

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0378337, upper bound: 1.0401547
time: 4.48 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0378338, upper bound: 1.0414155
time: 3.53 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 21.53 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 21.53
Output dim: 9, lower bound: -1.0388888, upper bound: 1.0334002
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 21.53
Output dim: 9, lower bound: -1.0401535, upper bound: 1.0378338
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 21.53
Output dim: 9, lower bound: -1.0378337, upper bound: 1.0401547
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 21.53
Output dim: 9, lower bound: -1.0378338, upper bound: 1.0414155

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -12.5768433, -9.5098438, -12.6115246, -9.4764118, -2.7124882, 2.8737986
1: -11.6988020, -9.2271595, -11.7188730, -9.1970444, -2.1989322, 2.1427596
2: -8.0686321, -6.2408857, -8.1226158, -6.2192740, -1.8169425, 1.8673091
3: -7.6618118, -5.2063136, -7.6763258, -5.1499887, -2.3498192, 2.2681811
4: -3.6554968, -1.3665140, -3.6543727, -1.3821797, -2.2733171, 2.2878587
5: -5.9022832, -3.8887348, -5.9403853, -3.8636541, -2.0386291, 2.0516505
6: -16.8771343, -13.8280115, -16.8786907, -13.8361683, -2.5677910, 2.4996064
7: -4.6478276, -2.3027351, -4.6293869, -2.2960060, -2.3518217, 2.2718735
8: -5.2099733, -2.9604917, -5.2091684, -2.9467831, -1.8117561, 1.8127580
9: 4.4352484, 5.9557619, 4.4294024, 5.9651985, -1.4091122, 1.4069183

Time for backsubstitution: 14.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0357124, upper bound: 1.0334015
time: 5.03 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0357125, upper bound: 1.0333998
time: 5.43 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -12.5779991, -9.5066519, -12.6220856, -9.4579248, -2.7290893, 2.8922062
1: -11.7001781, -9.2250204, -11.7314796, -9.1848755, -2.2128901, 2.1654096
2: -8.0719109, -6.2380009, -8.1453514, -6.2022753, -1.8371882, 1.8940499
3: -7.6701622, -5.2049522, -7.7196836, -5.1315060, -2.3761578, 2.2969449
4: -3.6570044, -1.3592873, -3.6744504, -1.3455455, -2.3114588, 2.3151631
5: -5.9032869, -3.8842707, -5.9499154, -3.8396616, -2.0636253, 2.0656447
6: -16.8792248, -13.8215885, -16.8996506, -13.8021469, -2.5907068, 2.5279336
7: -4.6579971, -2.2996969, -4.6814713, -2.2616956, -2.3963015, 2.3016779
8: -5.2140493, -2.9587946, -5.2304287, -2.9316592, -1.8309786, 1.8247979
9: 4.4324579, 5.9562163, 4.4102211, 5.9689026, -1.4186511, 1.4306414

Time for backsubstitution: 14.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0357124, upper bound: 1.0365760
time: 6.23 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0357125, upper bound: 1.0378355
time: 3.62 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -12.6260767, -9.4461374, -12.5779991, -9.5066519, -2.8967237, 2.7449822
1: -11.7355165, -9.1776304, -11.7001781, -9.2250204, -2.1631756, 2.2202718
2: -8.1623936, -6.1997938, -8.0719109, -6.2380009, -1.9089494, 1.8393383
3: -7.7232981, -5.1149902, -7.6701622, -5.2049522, -2.3042085, 2.3905346
4: -3.6771874, -1.3426166, -3.6570044, -1.3592873, -2.3179002, 2.3143878
5: -5.9543214, -3.8286192, -5.9032869, -3.8842707, -2.0700507, 2.0746677
6: -16.9029369, -13.7977180, -16.8792248, -13.8215885, -2.5288877, 2.6098850
7: -4.6868491, -2.2577415, -4.6579971, -2.2996969, -2.3329952, 2.4002557
8: -5.2317619, -2.9253788, -5.2140493, -2.9587946, -1.8325062, 1.8399487
9: 4.4055643, 5.9714341, 4.4324579, 5.9562163, -1.4313188, 1.4245541

Time for backsubstitution: 14.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5799
type: A, layer: 1, pos: 5799
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0365750, upper bound: 1.0357124
time: 3.32 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0378327, upper bound: 1.0401529
time: 4.10 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -12.6260767, -9.4461374, -12.6260767, -9.4461374, -2.9360151, 2.9360147
1: -11.7355165, -9.1776304, -11.7355165, -9.1776304, -2.2021341, 2.2021341
2: -8.1623936, -6.1997938, -8.1623936, -6.1997938, -1.9110990, 1.9110992
3: -7.7232981, -5.1149902, -7.7232981, -5.1149902, -2.3985672, 2.3985670
4: -3.6771874, -1.3426166, -3.6771874, -1.3426166, -2.3345709, 2.3345709
5: -5.9543214, -3.8286192, -5.9543214, -3.8286192, -2.1257021, 2.1257021
6: -16.9029369, -13.7977180, -16.9029369, -13.7977180, -2.6677151, 2.6677153
7: -4.6868491, -2.2577415, -4.6868491, -2.2577415, -2.4291077, 2.4291077
8: -5.2317619, -2.9253788, -5.2317619, -2.9253788, -1.8587062, 1.8587060
9: 4.4055643, 5.9714341, 4.4055643, 5.9714341, -1.4502144, 1.4502144

Time for backsubstitution: 14.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5799
type: B, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 5790
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0333985, upper bound: 1.0401476
time: 3.56 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0378325, upper bound: 1.0414140
time: 3.59 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.37 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 9, lower bound: -1.0357124, upper bound: 1.0334015
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 9, lower bound: -1.0357125, upper bound: 1.0333998
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 9, lower bound: -1.0357124, upper bound: 1.0365760
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 9, lower bound: -1.0357125, upper bound: 1.0378355
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 9, lower bound: -1.0365750, upper bound: 1.0357124
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 9, lower bound: -1.0378327, upper bound: 1.0401529
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 9, lower bound: -1.0333985, upper bound: 1.0401476
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.37
Output dim: 9, lower bound: -1.0378325, upper bound: 1.0414140

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -12.5695992, -9.5244799, -12.6115246, -9.4764118, -2.7035203, 2.8549390
1: -11.6908426, -9.2366714, -11.7188730, -9.1970444, -2.1876125, 2.1338787
2: -8.0497160, -6.2521429, -8.1226158, -6.2192740, -1.7983415, 1.8549523
3: -7.6269441, -5.2209206, -7.6763258, -5.1499887, -2.3158503, 2.2534857
4: -3.6373177, -1.3955841, -3.6543727, -1.3821797, -2.2551379, 2.2587886
5: -5.8948550, -3.9073162, -5.9403853, -3.8636541, -2.0312009, 2.0330691
6: -16.8591957, -13.8542118, -16.8786907, -13.8361683, -2.5476184, 2.4736183
7: -4.6072283, -2.3321939, -4.6293869, -2.2960060, -2.3112223, 2.2460115
8: -5.1929550, -2.9719386, -5.2091684, -2.9467831, -1.7973599, 1.7990439
9: 4.4472542, 5.9525614, 4.4294024, 5.9651985, -1.3972929, 1.3991516

Time for backsubstitution: 13.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0357090, upper bound: 1.0323835
time: 4.69 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0357090, upper bound: 1.0333965
time: 4.69 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.5779982, -9.5066538, -12.6115246, -9.4764118, -2.7137055, 2.8780627
1: -11.7001743, -9.2250233, -11.7188730, -9.1970444, -2.1989717, 2.1457458
2: -8.0719080, -6.2380018, -8.1226158, -6.2192740, -1.8194568, 1.8705959
3: -7.6701522, -5.2049541, -7.6763258, -5.1499887, -2.3549666, 2.2695799
4: -3.6570032, -1.3592918, -3.6543727, -1.3821797, -2.2748234, 2.2950809
5: -5.9032845, -3.8842745, -5.9403853, -3.8636541, -2.0396304, 2.0561109
6: -16.8792229, -13.8215942, -16.8786907, -13.8361683, -2.5704112, 2.5059781
7: -4.6579862, -2.2996984, -4.6293869, -2.2960060, -2.3619802, 2.2739055
8: -5.2140436, -2.9587955, -5.2091684, -2.9467831, -1.8161273, 1.8119969
9: 4.4324603, 5.9562154, 4.4294024, 5.9651985, -1.4127722, 1.4050033

Time for backsubstitution: 14.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0357091, upper bound: 1.0323839
time: 4.06 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0357091, upper bound: 1.0333966
time: 3.60 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -12.5695992, -9.5244799, -12.6220856, -9.4579248, -2.7221026, 2.8690772
1: -11.6908426, -9.2366714, -11.7314796, -9.1848755, -2.1998129, 2.1465042
2: -8.0497160, -6.2521429, -8.1453514, -6.2022753, -1.8156972, 1.8770981
3: -7.6269441, -5.2209206, -7.7196836, -5.1315060, -2.3340321, 2.2955937
4: -3.6373177, -1.3955841, -3.6744504, -1.3455455, -2.2917721, 2.2788663
5: -5.8948550, -3.9073162, -5.9499154, -3.8396616, -2.0551934, 2.0425992
6: -16.8591957, -13.8542118, -16.8996506, -13.8021469, -2.5810747, 2.4955678
7: -4.6072283, -2.3321939, -4.6814713, -2.2616956, -2.3455327, 2.2989564
8: -5.1929550, -2.9719386, -5.2304287, -2.9316592, -1.8122075, 1.8176880
9: 4.4472542, 5.9525614, 4.4102211, 5.9689026, -1.4031684, 1.4187171

Time for backsubstitution: 14.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0357089, upper bound: 1.0355599
time: 3.78 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0357089, upper bound: 1.0365727
time: 4.29 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.5779982, -9.5066538, -12.6220856, -9.4579248, -2.7287755, 2.8897436
1: -11.7001743, -9.2250233, -11.7314796, -9.1848755, -2.2202430, 2.1654067
2: -8.0719080, -6.2380018, -8.1453514, -6.2022753, -1.8387833, 1.8940473
3: -7.6701522, -5.2049541, -7.7196836, -5.1315060, -2.3609881, 2.2969432
4: -3.6570032, -1.3592918, -3.6744504, -1.3455455, -2.3114576, 2.3091888
5: -5.9032845, -3.8842745, -5.9499154, -3.8396616, -2.0636230, 2.0656409
6: -16.8792229, -13.8215942, -16.8996506, -13.8021469, -2.5907044, 2.5140104
7: -4.6579862, -2.2996984, -4.6814713, -2.2616956, -2.3802404, 2.3016763
8: -5.2140436, -2.9587955, -5.2304287, -2.9316592, -1.8259342, 1.8247945
9: 4.4324603, 5.9562154, 4.4102211, 5.9689026, -1.4251578, 1.4306393

Time for backsubstitution: 14.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0357090, upper bound: 1.0323831
time: 5.42 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0357090, upper bound: 1.0355942
time: 5.31 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -12.6245041, -9.4494190, -12.5695992, -9.5244799, -2.8712769, 2.7311435
1: -11.7335176, -9.1797867, -11.6908426, -9.2366714, -2.1491437, 2.2040679
2: -8.1590137, -6.2031250, -8.0497160, -6.2521429, -1.8902388, 1.8142049
3: -7.7149525, -5.1166658, -7.6269441, -5.2209206, -2.2712297, 2.3472109
4: -3.6755993, -1.3499100, -3.6373177, -1.3955841, -2.2800152, 2.2874076
5: -5.9531636, -3.8332615, -5.8948550, -3.9073162, -2.0458474, 2.0615935
6: -16.9006958, -13.8044062, -16.8591957, -13.8542118, -2.4941719, 2.5805156
7: -4.6764340, -2.2610009, -4.6072283, -2.3321939, -2.2945912, 2.3462274
8: -5.2277069, -2.9274507, -5.1929550, -2.9719386, -1.8127885, 1.8190794
9: 4.4091473, 5.9709649, 4.4472542, 5.9525614, -1.4183135, 1.4081752

Time for backsubstitution: 14.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5799
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0333994, upper bound: 1.0357120
time: 4.79 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0333994, upper bound: 1.0357140
time: 3.22 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -12.6260767, -9.4461374, -12.5779982, -9.5066538, -2.8928366, 2.7446687
1: -11.7355165, -9.1776304, -11.7001743, -9.2250233, -2.1631722, 2.2276249
2: -8.1623936, -6.1997938, -8.0719080, -6.2380018, -1.9089463, 1.8409338
3: -7.7232981, -5.1149902, -7.6701522, -5.2049541, -2.2959349, 2.3753726
4: -3.6771874, -1.3426166, -3.6570032, -1.3592918, -2.3050098, 2.3143866
5: -5.9543214, -3.8286192, -5.9032845, -3.8842745, -2.0700469, 2.0746653
6: -16.9029369, -13.7977180, -16.8792229, -13.8215942, -2.5149643, 2.6098828
7: -4.6868491, -2.2577415, -4.6579862, -2.2996984, -2.3329940, 2.3755534
8: -5.2317619, -2.9253788, -5.2140436, -2.9587955, -1.8300195, 1.8322687
9: 4.4055643, 5.9714341, 4.4324603, 5.9562154, -1.4285071, 1.4282633

Time for backsubstitution: 14.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5799
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0333994, upper bound: 1.0388906
time: 4.28 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0333992, upper bound: 1.0401554
time: 4.13 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.6155739, -9.4646454, -12.6245041, -9.4494190, -2.9160967, 2.9097366
1: -11.7230082, -9.1898232, -11.7335176, -9.1797867, -2.1851668, 2.1873965
2: -8.1397610, -6.2167883, -8.1590137, -6.2031250, -1.8859105, 1.8899341
3: -7.6798916, -5.1334405, -7.7149525, -5.1166658, -2.3541975, 2.3721831
4: -3.6571112, -1.3792863, -3.6755993, -1.3499100, -2.3072011, 2.2963130
5: -5.9448314, -3.8526349, -5.9531636, -3.8332615, -2.1115699, 2.1005287
6: -16.8820229, -13.8317642, -16.9006958, -13.8044062, -2.6366372, 2.6313670
7: -4.6346989, -2.2920003, -4.6764340, -2.2610009, -2.3736980, 2.3844337
8: -5.2104793, -2.9404683, -5.2277069, -2.9274507, -1.8377264, 1.8367736
9: 4.4248171, 5.9677439, 4.4091473, 5.9709649, -1.4297521, 1.4371802

Time for backsubstitution: 14.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 5790
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0363695, upper bound: 1.0369710
time: 10.49 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0363695, upper bound: 1.0401485
time: 4.11 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -12.6260729, -9.4461412, -12.6260767, -9.4461374, -2.9345369, 2.9321289
1: -11.7355156, -9.1776333, -11.7355165, -9.1776304, -2.2078362, 2.2021310
2: -8.1623917, -6.1997986, -8.1623936, -6.1997938, -1.9121552, 1.9110954
3: -7.7232900, -5.1149921, -7.7232981, -5.1149902, -2.3833961, 2.3985648
4: -3.6771855, -1.3426216, -3.6771874, -1.3426166, -2.3345690, 2.3345659
5: -5.9543195, -3.8286245, -5.9543214, -3.8286192, -2.1257002, 2.1256969
6: -16.9029350, -13.7977257, -16.9029369, -13.7977180, -2.6677117, 2.6545465
7: -4.6868372, -2.2577446, -4.6868491, -2.2577415, -2.4290957, 2.4291046
8: -5.2317557, -2.9253788, -5.2317619, -2.9253788, -1.8510723, 1.8559532
9: 4.4055681, 5.9714327, 4.4055643, 5.9714341, -1.4534590, 1.4473746

Time for backsubstitution: 14.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 5790
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0395459, upper bound: 1.0369740
time: 3.78 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0395459, upper bound: 1.0414145
time: 3.74 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.78 seconds
IS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 21.78
Output dim: 9, lower bound: -1.0357090, upper bound: 1.0323835
IS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 21.78
Output dim: 9, lower bound: -1.0357090, upper bound: 1.0333965
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 21.78
Output dim: 9, lower bound: -1.0357091, upper bound: 1.0323839
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 21.78
Output dim: 9, lower bound: -1.0357091, upper bound: 1.0333966
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 21.78
Output dim: 9, lower bound: -1.0357089, upper bound: 1.0355599
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 21.78
Output dim: 9, lower bound: -1.0357089, upper bound: 1.0365727
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 21.78
Output dim: 9, lower bound: -1.0357090, upper bound: 1.0323831
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 21.78
Output dim: 9, lower bound: -1.0357090, upper bound: 1.0355942
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 21.78
Output dim: 9, lower bound: -1.0333994, upper bound: 1.0357120
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 21.78
Output dim: 9, lower bound: -1.0333994, upper bound: 1.0357140
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 21.78
Output dim: 9, lower bound: -1.0333994, upper bound: 1.0388906
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 21.78
Output dim: 9, lower bound: -1.0333992, upper bound: 1.0401554
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 21.78
Output dim: 9, lower bound: -1.0363695, upper bound: 1.0369710
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 21.78
Output dim: 9, lower bound: -1.0363695, upper bound: 1.0401485
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 21.78
Output dim: 9, lower bound: -1.0395459, upper bound: 1.0369740
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 21.78
Output dim: 9, lower bound: -1.0395459, upper bound: 1.0414145

## BFS IS instance: IS_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -12.5557985, -9.5246735, -12.6099577, -9.4764366, -2.6902146, 2.8530498
1: -11.6890669, -9.2372208, -11.7186165, -9.1971006, -2.1843071, 2.1312616
2: -8.0416355, -6.2525535, -8.1217251, -6.2193513, -1.7898285, 1.8536952
3: -7.6172152, -5.2212663, -7.6752701, -5.1500621, -2.3056350, 2.2517917
4: -3.6367331, -1.3977122, -3.6542969, -1.3824155, -2.2543175, 2.2565846
5: -5.8909998, -3.9076686, -5.9399385, -3.8637028, -2.0272970, 2.0322700
6: -16.8495903, -13.8546124, -16.8776360, -13.8362446, -2.5381231, 2.4723718
7: -4.6064076, -2.3327527, -4.6292825, -2.2960997, -2.3103080, 2.2452421
8: -5.1925311, -2.9749517, -5.2091193, -2.9471407, -1.7961512, 1.7954247
9: 4.4476795, 5.9510565, 4.4295030, 5.9650345, -1.3959382, 1.3965030

Time for backsubstitution: 14.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A1_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0346806, upper bound: 1.0323834
time: 4.17 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0346806, upper bound: 1.0323839
time: 3.89 seconds

## BFS IS instance: IS_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -12.5722809, -9.5089931, -12.6115112, -9.4764090, -2.7044988, 2.8703094
1: -11.6922483, -9.2333393, -11.7188702, -9.1970444, -2.1905541, 2.1356189
2: -8.0506458, -6.2421389, -8.1226110, -6.2192740, -1.7980139, 1.8647687
3: -7.6287971, -5.2071543, -7.6763220, -5.1499901, -2.3157306, 2.2668047
4: -3.6410840, -1.3947358, -3.6543725, -1.3821816, -2.2589023, 2.2596366
5: -5.8962679, -3.9012523, -5.9403849, -3.8636551, -2.0326128, 2.0391326
6: -16.8610611, -13.8423557, -16.8786850, -13.8361702, -2.5481091, 2.4854739
7: -4.6093516, -2.3291690, -4.6293874, -2.2960064, -2.3133452, 2.2487795
8: -5.1970825, -2.9691768, -5.2091665, -2.9467854, -1.8012364, 1.8023713
9: 4.4441676, 5.9529352, 4.4294038, 5.9651961, -1.3996179, 1.4002609

Time for backsubstitution: 14.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0346806, upper bound: 1.0333962
time: 4.37 seconds

## Relational analysis of IS_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0346806, upper bound: 1.0333968
time: 3.67 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -12.5642014, -9.5068512, -12.6099577, -9.4764366, -2.7004046, 2.8761764
1: -11.6983986, -9.2255669, -11.7186165, -9.1971006, -2.1956606, 2.1431444
2: -8.0638199, -6.2384143, -8.1217251, -6.2193513, -1.8109467, 1.8693383
3: -7.6604395, -5.2053022, -7.6752701, -5.1500621, -2.3447568, 2.2678869
4: -3.6564171, -1.3614161, -3.6542969, -1.3824155, -2.2740016, 2.2928808
5: -5.8994303, -3.8846354, -5.9399385, -3.8637028, -2.0357275, 2.0553031
6: -16.8696213, -13.8219967, -16.8776360, -13.8362446, -2.5609217, 2.5047319
7: -4.6571789, -2.3002496, -4.6292825, -2.2960997, -2.3610792, 2.2731287
8: -5.2136230, -2.9617991, -5.2091193, -2.9471407, -1.8149319, 1.8083720
9: 4.4328890, 5.9547091, 4.4295030, 5.9650345, -1.4114161, 1.4023519

Time for backsubstitution: 14.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0378568, upper bound: 1.0323853
time: 3.66 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0378568, upper bound: 1.0323836
time: 3.93 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -12.5806799, -9.4911633, -12.6115112, -9.4764090, -2.7146978, 2.8934426
1: -11.7015696, -9.2216749, -11.7188702, -9.1970444, -2.2019200, 2.1475182
2: -8.0728168, -6.2279987, -8.1226110, -6.2192740, -1.8191202, 1.8804109
3: -7.6720057, -5.1911883, -7.6763220, -5.1499901, -2.3548899, 2.2796934
4: -3.6607685, -1.3584409, -3.6543725, -1.3821816, -2.2785869, 2.2959316
5: -5.9046988, -3.8781958, -5.9403849, -3.8636551, -2.0410438, 2.0621891
6: -16.8810902, -13.8097363, -16.8786850, -13.8361702, -2.5709057, 2.5178347
7: -4.6601005, -2.2966342, -4.6293874, -2.2960064, -2.3640940, 2.2766783
8: -5.2181773, -2.9560452, -5.2091665, -2.9467854, -1.8200190, 1.8153250
9: 4.4293709, 5.9565854, 4.4294038, 5.9651961, -1.4151042, 1.4061112

Time for backsubstitution: 14.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4628

## Relational analysis of IS_A1_B1_A2_A2_A1

### Relational analysis result of IS_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0357158, upper bound: 1.0333916
time: 3.60 seconds

## Relational analysis of IS_A1_B1_A2_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0388788, upper bound: 1.0333916
time: 3.59 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -12.5557985, -9.5246735, -12.6205235, -9.4579525, -2.7087960, 2.8671880
1: -11.6890669, -9.2372208, -11.7312260, -9.1849337, -2.1965075, 2.1438873
2: -8.0416355, -6.2525535, -8.1444607, -6.2023525, -1.8071845, 1.8758399
3: -7.6172152, -5.2212663, -7.7186289, -5.1315784, -2.3238177, 2.2939036
4: -3.6367331, -1.3977122, -3.6743755, -1.3457799, -2.2909532, 2.2766633
5: -5.8909998, -3.9076686, -5.9494677, -3.8397121, -2.0512877, 2.0417991
6: -16.8495903, -13.8546124, -16.8985939, -13.8022232, -2.5715799, 2.4943213
7: -4.6064076, -2.3327527, -4.6813669, -2.2617908, -2.3446169, 2.2981875
8: -5.1925311, -2.9749517, -5.2303829, -2.9320159, -1.8109994, 1.8140695
9: 4.4476795, 5.9510565, 4.4103222, 5.9687405, -1.4018126, 1.4160688

Time for backsubstitution: 14.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0346808, upper bound: 1.0355615
time: 3.71 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0346806, upper bound: 1.0355598
time: 3.65 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -12.5722809, -9.5089931, -12.6220760, -9.4579239, -2.7230816, 2.8844476
1: -11.6922483, -9.2333393, -11.7314796, -9.1848774, -2.2027555, 2.1482444
2: -8.0506458, -6.2421389, -8.1453476, -6.2022767, -1.8153696, 1.8869138
3: -7.6287971, -5.2071543, -7.7196779, -5.1315079, -2.3339119, 2.2989869
4: -3.6410840, -1.3947358, -3.6744502, -1.3455482, -2.2955358, 2.2797143
5: -5.8962679, -3.9012523, -5.9499121, -3.8396628, -2.0566051, 2.0486598
6: -16.8610611, -13.8423557, -16.8996468, -13.8021507, -2.5815654, 2.5074234
7: -4.6093516, -2.3291690, -4.6814713, -2.2616966, -2.3476551, 2.3017237
8: -5.1970825, -2.9691768, -5.2304306, -2.9316597, -1.8160839, 1.8210146
9: 4.4441676, 5.9529352, 4.4102221, 5.9689026, -1.4054928, 1.4198277

Time for backsubstitution: 14.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4628

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0357026, upper bound: 1.0334050
time: 3.83 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0357028, upper bound: 1.0365680
time: 3.64 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -12.5642014, -9.5068512, -12.6205235, -9.4579525, -2.7154746, 2.8878567
1: -11.6983986, -9.2255669, -11.7312260, -9.1849337, -2.2169318, 2.1628053
2: -8.0638199, -6.2384143, -8.1444607, -6.2023525, -1.8302734, 1.8927891
3: -7.6604395, -5.2053022, -7.7186289, -5.1315784, -2.3507924, 2.2952538
4: -3.6564171, -1.3614161, -3.6743755, -1.3457799, -2.3106372, 2.3068180
5: -5.8994303, -3.8846354, -5.9494677, -3.8397121, -2.0597181, 2.0648322
6: -16.8696213, -13.8219967, -16.8985939, -13.8022232, -2.5812163, 2.5127635
7: -4.6571789, -2.3002496, -4.6813669, -2.2617908, -2.3787112, 2.3009000
8: -5.2136230, -2.9617991, -5.2303829, -2.9320159, -1.8247147, 1.8211672
9: 4.4328890, 5.9547091, 4.4103222, 5.9687405, -1.4238040, 1.4279953

Time for backsubstitution: 13.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A1_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0391215, upper bound: 1.0345813
time: 4.73 seconds

## Relational analysis of IS_A1_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0391216, upper bound: 1.0345832
time: 4.13 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -12.5806799, -9.4911633, -12.6220760, -9.4579239, -2.7297683, 2.9051249
1: -11.7015696, -9.2216749, -11.7314796, -9.1848774, -2.2231922, 2.1671789
2: -8.0728168, -6.2279987, -8.1453476, -6.2022767, -1.8384461, 1.9038620
3: -7.6720057, -5.1911883, -7.7196779, -5.1315079, -2.3608885, 2.3102634
4: -3.6607685, -1.3584409, -3.6744502, -1.3455482, -2.3152204, 2.3100204
5: -5.9046988, -3.8781958, -5.9499121, -3.8396628, -2.0650361, 2.0717163
6: -16.8810902, -13.8097363, -16.8996468, -13.8021507, -2.5912001, 2.5258679
7: -4.6601005, -2.2966342, -4.6814713, -2.2616966, -2.3811669, 2.3044488
8: -5.2181773, -2.9560452, -5.2304306, -2.9316597, -1.8298078, 1.8281391
9: 4.4293709, 5.9565854, 4.4102221, 5.9689026, -1.4274857, 1.4317534

Time for backsubstitution: 14.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4628

## Relational analysis of IS_A1_B2_A2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0369806, upper bound: 1.0355898
time: 3.70 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0401435, upper bound: 1.0355900
time: 3.61 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -12.6155739, -9.4646454, -12.5695992, -9.5244799, -2.8580270, 2.7159324
1: -11.7230082, -9.1898232, -11.6908426, -9.2366714, -2.1373515, 2.1949687
2: -8.1397610, -6.2167883, -8.0497160, -6.2521429, -1.8710976, 1.8004076
3: -7.6798916, -5.1334405, -7.6269441, -5.2209206, -2.2460303, 2.3309622
4: -3.6571112, -1.3792863, -3.6373177, -1.3955841, -2.2615271, 2.2580314
5: -5.9448314, -3.8526349, -5.8948550, -3.9073162, -2.0375152, 2.0422201
6: -16.8820229, -13.8317642, -16.8591957, -13.8542118, -2.4748862, 2.5536144
7: -4.6346989, -2.2920003, -4.6072283, -2.3321939, -2.2521060, 2.3152280
8: -5.2104793, -2.9404683, -5.1929550, -2.9719386, -1.7983899, 1.8036189
9: 4.4248171, 5.9677439, 4.4472542, 5.9525614, -1.4030893, 1.4003816

Time for backsubstitution: 14.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0323831, upper bound: 1.0357084
time: 4.69 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0333960, upper bound: 1.0357106
time: 4.32 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -12.6260729, -9.4461412, -12.5695992, -9.5244799, -2.8721156, 2.7344804
1: -11.7355156, -9.1776333, -11.6908426, -9.2366714, -2.1499729, 2.2071898
2: -8.1623917, -6.1997986, -8.0497160, -6.2521429, -1.8930531, 1.8178437
3: -7.7232900, -5.1149921, -7.6269441, -5.2209206, -2.2732105, 2.3404858
4: -3.6771855, -1.3426216, -3.6373177, -1.3955841, -2.2816014, 2.2946961
5: -5.9543195, -3.8286245, -5.8948550, -3.9073162, -2.0470033, 2.0662305
6: -16.9029350, -13.7977257, -16.8591957, -13.8542118, -2.4965189, 2.5870843
7: -4.6868372, -2.2577446, -4.6072283, -2.3321939, -2.3050926, 2.3494837
8: -5.2317557, -2.9253788, -5.1929550, -2.9719386, -1.8170619, 1.8184247
9: 4.4055681, 5.9714327, 4.4472542, 5.9525614, -1.4226513, 1.4062312

Time for backsubstitution: 13.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0323831, upper bound: 1.0357084
time: 7.10 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0333960, upper bound: 1.0357084
time: 4.73 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -12.6155739, -9.4646454, -12.5779982, -9.5066538, -2.8811512, 2.7261178
1: -11.7230082, -9.1898232, -11.7001743, -9.2250233, -2.1492181, 2.2063279
2: -8.1397610, -6.2167883, -8.0719080, -6.2380018, -1.8867409, 1.8215232
3: -7.6798916, -5.1334405, -7.6701522, -5.2049541, -2.2535851, 2.3577151
4: -3.6571112, -1.3792863, -3.6570032, -1.3592918, -2.2978194, 2.2777169
5: -5.9448314, -3.8526349, -5.9032845, -3.8842745, -2.0605569, 2.0506496
6: -16.8820229, -13.8317642, -16.8792229, -13.8215942, -2.5072451, 2.5764072
7: -4.6346989, -2.2920003, -4.6579862, -2.2996984, -2.2799997, 2.3659859
8: -5.2104793, -2.9404683, -5.2140436, -2.9587955, -1.8113427, 1.8223865
9: 4.4248171, 5.9677439, 4.4324603, 5.9562154, -1.4089413, 1.4158607

Time for backsubstitution: 14.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0323829, upper bound: 1.0388848
time: 4.60 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0333958, upper bound: 1.0388845
time: 4.55 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -12.6260729, -9.4461412, -12.5779982, -9.5066538, -2.8928328, 2.7411532
1: -11.7355156, -9.1776333, -11.7001743, -9.2250233, -2.1688747, 2.2276206
2: -8.1623917, -6.1997986, -8.0719080, -6.2380018, -1.9100025, 1.8409300
3: -7.7232900, -5.1149921, -7.6701522, -5.2049541, -2.2873309, 2.3734560
4: -3.6771855, -1.3426216, -3.6570032, -1.3592918, -2.3050089, 2.3143816
5: -5.9543195, -3.8286245, -5.9032845, -3.8842745, -2.0700450, 2.0746601
6: -16.9029350, -13.7977257, -16.8792229, -13.8215942, -2.5149620, 2.5967145
7: -4.6868372, -2.2577446, -4.6579862, -2.2996984, -2.3078125, 2.3755515
8: -5.2317557, -2.9253788, -5.2140436, -2.9587955, -1.8241186, 1.8322647
9: 4.4055681, 5.9714327, 4.4324603, 5.9562154, -1.4344747, 1.4282612

Time for backsubstitution: 14.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0323830, upper bound: 1.0357113
time: 3.05 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0333959, upper bound: 1.0379151
time: 3.47 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -12.6155739, -9.4646454, -12.6155739, -9.4646454, -2.8964872, 2.8964872
1: -11.7230082, -9.1898232, -11.7230082, -9.1898232, -2.1756039, 2.1756041
2: -8.1397610, -6.2167883, -8.1397610, -6.2167883, -1.8713360, 1.8713360
3: -7.6798916, -5.1334405, -7.6798916, -5.1334405, -2.3376546, 2.3376548
4: -3.6571112, -1.3792863, -3.6571112, -1.3792863, -2.2778249, 2.2778249
5: -5.9448314, -3.8526349, -5.9448314, -3.8526349, -2.0921965, 2.0921965
6: -16.8820229, -13.8317642, -16.8820229, -13.8317642, -2.6098328, 2.6098328
7: -4.6346989, -2.2920003, -4.6346989, -2.2920003, -2.3426986, 2.3426986
8: -5.2104793, -2.9404683, -5.2104793, -2.9404683, -1.8222656, 1.8222659
9: 4.4248171, 5.9677439, 4.4248171, 5.9677439, -1.4219561, 1.4219559

Time for backsubstitution: 14.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 5860

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0363661, upper bound: 1.0359359
time: 4.60 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0363660, upper bound: 1.0369681
time: 4.85 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -12.6155739, -9.4646454, -12.6260729, -9.4461412, -2.9204435, 2.9105754
1: -11.7230082, -9.1898232, -11.7355156, -9.1776333, -2.1881766, 2.1882250
2: -8.1397610, -6.2167883, -8.1623917, -6.1997986, -1.8896208, 1.8925614
3: -7.6798916, -5.1334405, -7.7232900, -5.1149921, -2.3557763, 2.3804374
4: -3.6571112, -1.3792863, -3.6771855, -1.3426216, -2.3144896, 2.2978992
5: -5.9448314, -3.8526349, -5.9543195, -3.8286245, -2.1162069, 2.1016846
6: -16.8820229, -13.8317642, -16.9029350, -13.7977257, -2.6431808, 2.6343582
7: -4.6346989, -2.2920003, -4.6868372, -2.2577446, -2.3769543, 2.3948369
8: -5.2104793, -2.9404683, -5.2317557, -2.9253788, -1.8370719, 1.8411434
9: 4.4248171, 5.9677439, 4.4055681, 5.9714327, -1.4278085, 1.4415179

Time for backsubstitution: 14.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 5790
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0353331, upper bound: 1.0365720
time: 4.77 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0363661, upper bound: 1.0401450
time: 4.19 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -12.6260729, -9.4461412, -12.6155739, -9.4646454, -2.9105759, 2.9204431
1: -11.7355156, -9.1776333, -11.7230082, -9.1898232, -2.1882253, 2.1881769
2: -8.1623917, -6.1997986, -8.1397610, -6.2167883, -1.8925610, 1.8896208
3: -7.7232900, -5.1149921, -7.6798916, -5.1334405, -2.3804374, 2.3557758
4: -3.6771855, -1.3426216, -3.6571112, -1.3792863, -2.2978992, 2.3144896
5: -5.9543195, -3.8286245, -5.9448314, -3.8526349, -2.1016846, 2.1162069
6: -16.9029350, -13.7977257, -16.8820229, -13.8317642, -2.6343584, 2.6431806
7: -4.6868372, -2.2577446, -4.6346989, -2.2920003, -2.3948369, 2.3769543
8: -5.2317557, -2.9253788, -5.2104793, -2.9404683, -1.8411436, 1.8370714
9: 4.4055681, 5.9714327, 4.4248171, 5.9677439, -1.4415181, 1.4278086

Time for backsubstitution: 14.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 5860

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0395423, upper bound: 1.0359355
time: 6.32 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0395424, upper bound: 1.0369692
time: 3.56 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -12.6260729, -9.4461412, -12.6260729, -9.4461412, -2.9321256, 2.9321256
1: -11.7355156, -9.1776333, -11.7355156, -9.1776333, -2.2078333, 2.2078333
2: -8.1623917, -6.1997986, -8.1623917, -6.1997986, -1.9121513, 1.9121513
3: -7.7232900, -5.1149921, -7.7232900, -5.1149921, -2.3833947, 2.3833942
4: -3.6771855, -1.3426216, -3.6771855, -1.3426216, -2.3345640, 2.3345640
5: -5.9543195, -3.8286245, -5.9543195, -3.8286245, -2.1256950, 2.1256950
6: -16.9029350, -13.7977257, -16.9029350, -13.7977257, -2.6545436, 2.6545436
7: -4.6868372, -2.2577446, -4.6868372, -2.2577446, -2.4290926, 2.4290926
8: -5.2317557, -2.9253788, -5.2317557, -2.9253788, -1.8510690, 1.8510690
9: 4.4055681, 5.9714327, 4.4055681, 5.9714327, -1.4534571, 1.4534571

Time for backsubstitution: 14.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 5790
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 5860

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0395425, upper bound: 1.0381459
time: 3.59 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0395425, upper bound: 1.0369687
time: 4.05 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 21.92 seconds
IS_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0346806, upper bound: 1.0323834
IS_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0346806, upper bound: 1.0323839
IS_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0346806, upper bound: 1.0333962
IS_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0346806, upper bound: 1.0333968
IS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0378568, upper bound: 1.0323853
IS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0378568, upper bound: 1.0323836
IS_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0357158, upper bound: 1.0333916
IS_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0388788, upper bound: 1.0333916
IS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0346808, upper bound: 1.0355615
IS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0346806, upper bound: 1.0355598
IS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0357026, upper bound: 1.0334050
IS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0357028, upper bound: 1.0365680
IS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0391215, upper bound: 1.0345813
IS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0391216, upper bound: 1.0345832
IS_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0369806, upper bound: 1.0355898
IS_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0401435, upper bound: 1.0355900
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0323831, upper bound: 1.0357084
IS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0333960, upper bound: 1.0357106
IS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0323831, upper bound: 1.0357084
IS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0333960, upper bound: 1.0357084
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0323829, upper bound: 1.0388848
IS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0333958, upper bound: 1.0388845
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0323830, upper bound: 1.0357113
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0333959, upper bound: 1.0379151
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0363661, upper bound: 1.0359359
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0363660, upper bound: 1.0369681
IS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0353331, upper bound: 1.0365720
IS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0363661, upper bound: 1.0401450
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0395423, upper bound: 1.0359355
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0395424, upper bound: 1.0369692
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0395425, upper bound: 1.0381459
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 21.92
Output dim: 9, lower bound: -1.0395425, upper bound: 1.0369687

## BFS IS instance: IS_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -12.5557985, -9.5246735, -12.5971537, -9.4766531, -2.6900315, 2.8402534
1: -11.6890669, -9.2372208, -11.7165298, -9.1975651, -2.1829815, 2.1277230
2: -8.0416355, -6.2525535, -8.1144094, -6.2199874, -1.7893267, 1.8460562
3: -7.6172152, -5.2212663, -7.6666031, -5.1506624, -2.3048697, 2.2428677
4: -3.6367331, -1.3977122, -3.6536937, -1.3843184, -2.2524147, 2.2559814
5: -5.8909998, -3.9076686, -5.9362826, -3.8641119, -2.0268879, 2.0286140
6: -16.8495903, -13.8546124, -16.8689423, -13.8368626, -2.5378418, 2.4636300
7: -4.6064076, -2.3327527, -4.6284423, -2.2968829, -2.3095248, 2.2441292
8: -5.1925311, -2.9749517, -5.2087154, -2.9500480, -1.7926121, 1.7948020
9: 4.4476795, 5.9510565, 4.4303293, 5.9637032, -1.3937390, 1.3951131

Time for backsubstitution: 14.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4628

## Relational analysis of IS_A1_B1_A1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0346813, upper bound: 1.0292159
time: 3.71 seconds

## Relational analysis of IS_A1_B1_A1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0346810, upper bound: 1.0323788
time: 3.60 seconds

## BFS IS instance: IS_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -12.5557985, -9.5246735, -12.6142244, -9.4609098, -2.7057962, 2.8571706
1: -11.6890669, -9.2372208, -11.7203865, -9.1936989, -2.1867867, 2.1320238
2: -8.0416355, -6.2525535, -8.1235676, -6.2092452, -1.7997475, 1.8555446
3: -7.6172152, -5.2212663, -7.6781425, -5.1361985, -2.3190036, 2.2541225
4: -3.6367331, -1.3977122, -3.6581421, -1.3813138, -2.2554193, 2.2604299
5: -5.8909998, -3.9076686, -5.9417877, -3.8575618, -2.0334380, 2.0341191
6: -16.8495903, -13.8546124, -16.8805695, -13.8243198, -2.5500212, 2.4755619
7: -4.6064076, -2.3327527, -4.6315169, -2.2929709, -2.3134367, 2.2467349
8: -5.1925311, -2.9749517, -5.2132721, -2.9439421, -1.7989106, 1.7994618
9: 4.4476795, 5.9510565, 4.4262028, 5.9655781, -1.3958688, 1.3991030

Time for backsubstitution: 13.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4628

## Relational analysis of IS_A1_B1_A1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0346813, upper bound: 1.0292141
time: 4.88 seconds

## Relational analysis of IS_A1_B1_A1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -1.0346810, upper bound: 1.0323769
time: 3.97 seconds

## BFS IS instance: IS_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -12.5722809, -9.5089931, -12.5971537, -9.4766531, -2.7065859, 2.8559623
1: -11.6922483, -9.2333393, -11.7165298, -9.1975651, -2.1866055, 2.1316512
2: -8.0506458, -6.2421389, -8.1144094, -6.2199874, -1.7986422, 1.8562038
3: -7.6287971, -5.2071543, -7.6666031, -5.1506624, -2.3162317, 2.2567964
4: -3.6410840, -1.3947358, -3.6536937, -1.3843184, -2.2567656, 2.2589579
5: -5.8962679, -3.9012523, -5.9362826, -3.8641119, -2.0321560, 2.0350304
6: -16.8610611, -13.8423557, -16.8689423, -13.8368626, -2.5494046, 2.4756691
7: -4.6093516, -2.3291690, -4.6284423, -2.2968829, -2.3124688, 2.2475307
8: -5.1970825, -2.9691768, -5.2087154, -2.9500480, -1.7972684, 1.8007638
9: 4.4441676, 5.9529352, 4.4303293, 5.9637032, -1.3971519, 1.3972508

Time for backsubstitution: 13.83 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=6, k_high=12, k_mid=9, eps_mid=0.0351562, abs_max=1.4499115943908691
rel_dist={9: [-1.041417153698955, 1.0414178245879198]}

## Binary search (step 1) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6166
type: B, layer: 1, pos: 6166
type: A, layer: 1, pos: 5799
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 6166

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9245890, upper bound: 0.9226015
time: 3.38 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9264223, upper bound: 0.9264247
time: 4.11 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 7.69 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 7.69
Output dim: 9, lower bound: -0.9245890, upper bound: 0.9226015
IS_A2, status: Status.UNKNOWN, split count: 1, time: 7.69
Output dim: 9, lower bound: -0.9264223, upper bound: 0.9264247

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -12.5779991, -9.5066519, -12.6202202, -9.4634037, -2.4837885, 2.6433346
1: -11.7001781, -9.2250204, -11.7295828, -9.1882238, -2.0443482, 1.9973972
2: -8.0719109, -6.2380009, -8.1374769, -6.2034402, -1.6914310, 1.7392704
3: -7.6701622, -5.2049522, -7.7179928, -5.1391630, -2.2026434, 2.1493747
4: -3.6570044, -1.3592873, -3.6731925, -1.3468895, -2.2028322, 2.1938848
5: -5.9032869, -3.8842707, -5.9478464, -3.8447785, -1.9694443, 1.9843340
6: -16.8792248, -13.8215885, -16.8981075, -13.8041916, -2.3373446, 2.2642081
7: -4.6579971, -2.2996969, -4.6790051, -2.2635503, -2.2310848, 2.1452134
8: -5.2140493, -2.9587946, -5.2298098, -2.9345465, -1.6650934, 1.6671289
9: 4.4324579, 5.9562163, 4.4123602, 5.9677401, -1.3239775, 1.3293986

Time for backsubstitution: 14.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5799
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9233006, upper bound: 0.9188837
time: 3.32 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9245880, upper bound: 0.9226004
time: 3.21 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -12.6260767, -9.4461374, -12.6260834, -9.4460955, -2.7095714, 2.6841397
1: -11.7355165, -9.1776304, -11.7355261, -9.1776152, -2.0463634, 2.0408134
2: -8.1623936, -6.1997938, -8.1624203, -6.1997881, -1.7595816, 1.8019509
3: -7.7232981, -5.1149902, -7.7233047, -5.1149726, -2.2731490, 2.2245097
4: -3.6771874, -1.3426166, -3.6771939, -1.3426106, -2.2384577, 2.2133625
5: -5.9543214, -3.8286192, -5.9543295, -3.8286088, -2.0483255, 2.0245275
6: -16.9029369, -13.7977180, -16.9029427, -13.7977076, -2.3777719, 2.4005306
7: -4.6868491, -2.2577415, -4.6868634, -2.2577329, -2.2661886, 2.2883806
8: -5.2317619, -2.9253788, -5.2317648, -2.9253664, -1.6943936, 1.6927810
9: 4.4055643, 5.9714341, 4.4055533, 5.9714379, -1.3541093, 1.3538069

Time for backsubstitution: 13.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5799
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 6166
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790
type: A, layer: 1, pos: 5860

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9227771, upper bound: 0.9251480
time: 3.50 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9264212, upper bound: 0.9264214
time: 6.09 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 23.18 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 23.18
Output dim: 9, lower bound: -0.9233006, upper bound: 0.9188837
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 23.18
Output dim: 9, lower bound: -0.9245880, upper bound: 0.9226004
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 23.18
Output dim: 9, lower bound: -0.9227771, upper bound: 0.9251480
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 23.18
Output dim: 9, lower bound: -0.9264212, upper bound: 0.9264214

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -12.5763569, -9.5111628, -12.6096344, -9.4818821, -2.4630141, 2.6216812
1: -11.6982298, -9.2280436, -11.7169247, -9.2003841, -2.0297918, 1.9793539
2: -8.0672626, -6.2420788, -8.1147022, -6.2204437, -1.6699829, 1.7121310
3: -7.6583548, -5.2068830, -7.6746449, -5.1576643, -2.1723258, 2.1054523
4: -3.6548641, -1.3695045, -3.6531134, -1.3835120, -2.1634464, 2.1653988
5: -5.9018660, -3.8905776, -5.9382973, -3.8687656, -1.9438362, 1.9576113
6: -16.8762646, -13.8306608, -16.8771267, -13.8382072, -2.3002889, 2.2332010
7: -4.6436148, -2.3040099, -4.6269360, -2.2978885, -2.1866493, 2.0893946
8: -5.2082896, -2.9611993, -5.2085495, -2.9496889, -1.6413186, 1.6460490
9: 4.4363756, 5.9555745, 4.4315367, 5.9640265, -1.3100731, 1.3085515

Time for backsubstitution: 14.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9232981, upper bound: 0.9181072
time: 4.00 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9232981, upper bound: 0.9188830
time: 3.22 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -12.5779991, -9.5066519, -12.6202183, -9.4634056, -2.4798160, 2.6418736
1: -11.7001781, -9.2250204, -11.7295809, -9.1882257, -2.0443449, 2.0026727
2: -8.0719109, -6.2380009, -8.1374731, -6.2034431, -1.6914268, 1.7402465
3: -7.6701622, -5.2049522, -7.7179842, -5.1391625, -2.2026415, 2.1326597
4: -3.6570044, -1.3592873, -3.6731915, -1.3468938, -2.1878748, 2.1938832
5: -5.9032869, -3.8842707, -5.9478450, -3.8447824, -1.9668260, 1.9787321
6: -16.8792248, -13.8215885, -16.8981056, -13.8041973, -2.3224277, 2.2642062
7: -4.6579971, -2.2996969, -4.6789947, -2.2635541, -2.2310824, 2.1166742
8: -5.2140493, -2.9587946, -5.2298040, -2.9345465, -1.6623814, 1.6570621
9: 4.4324579, 5.9562163, 4.4123626, 5.9677410, -1.3211520, 1.3316772

Time for backsubstitution: 14.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9208665, upper bound: 0.9213131
time: 3.59 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9208665, upper bound: 0.9226005
time: 3.51 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -12.6155739, -9.4646454, -12.6238527, -9.4507322, -2.6878581, 2.6568904
1: -11.7230082, -9.1898232, -11.7326918, -9.1806602, -2.0281901, 2.0252242
2: -8.1397610, -6.2167883, -8.1576300, -6.2044973, -1.7328572, 1.7793188
3: -7.6798916, -5.1334405, -7.7115059, -5.1173506, -2.2286391, 2.1947076
4: -3.6571112, -1.3792863, -3.6749403, -1.3529205, -2.2098980, 2.1745470
5: -5.9448314, -3.8526349, -5.9526901, -3.8351660, -2.0215044, 1.9969854
6: -16.8820229, -13.8317642, -16.8997669, -13.8071566, -2.3445067, 2.3629365
7: -4.6346989, -2.2920003, -4.6721325, -2.2623580, -2.2101855, 2.2417400
8: -5.2104793, -2.9404683, -5.2260356, -2.9282999, -1.6725397, 1.6690354
9: 4.4248171, 5.9677439, 4.4106121, 5.9707747, -1.3332725, 1.3389606

Time for backsubstitution: 14.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5860

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6166

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9188824, upper bound: 0.9233030
time: 3.35 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9188824, upper bound: 0.9251457
time: 3.61 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -12.6260729, -9.4461412, -12.6260834, -9.4460955, -2.7080927, 2.6795745
1: -11.7355156, -9.1776333, -11.7355261, -9.1776152, -2.0516386, 2.0408103
2: -8.1623917, -6.1997986, -8.1624203, -6.1997881, -1.7605591, 1.8019464
3: -7.7232900, -5.1149921, -7.7233047, -5.1149726, -2.2559619, 2.2245073
4: -3.6771855, -1.3426216, -3.6771939, -1.3426106, -2.2384562, 2.1981559
5: -5.9543195, -3.8286245, -5.9543295, -3.8286088, -2.0426989, 2.0228698
6: -16.9029350, -13.7977257, -16.9029427, -13.7977076, -2.3777680, 2.3856137
7: -4.6868372, -2.2577446, -4.6868634, -2.2577329, -2.2364750, 2.2877078
8: -5.2317557, -2.9253788, -5.2317648, -2.9253664, -1.6849689, 1.6900280
9: 4.4055681, 5.9714327, 4.4055533, 5.9714379, -1.3563285, 1.3509675

Time for backsubstitution: 14.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 5790
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5860

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6166

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9225972, upper bound: 0.9245905
time: 3.50 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9225972, upper bound: 0.9264239
time: 3.58 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 21.32 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 21.32
Output dim: 9, lower bound: -0.9232981, upper bound: 0.9181072
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 21.32
Output dim: 9, lower bound: -0.9232981, upper bound: 0.9188830
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 21.32
Output dim: 9, lower bound: -0.9208665, upper bound: 0.9213131
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 21.32
Output dim: 9, lower bound: -0.9208665, upper bound: 0.9226005
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 21.32
Output dim: 9, lower bound: -0.9188824, upper bound: 0.9233030
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 21.32
Output dim: 9, lower bound: -0.9188824, upper bound: 0.9251457
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 21.32
Output dim: 9, lower bound: -0.9225972, upper bound: 0.9245905
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 21.32
Output dim: 9, lower bound: -0.9225972, upper bound: 0.9264239

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -12.5625601, -9.5113621, -12.6062183, -9.4819422, -2.4496856, 2.6179490
1: -11.6964588, -9.2285872, -11.7163677, -9.2005081, -2.0262747, 1.9762251
2: -8.0591784, -6.2424893, -8.1127548, -6.2206130, -1.6614003, 1.7097707
3: -7.6486378, -5.2072296, -7.6723385, -5.1578236, -2.1620038, 2.1024673
4: -3.6542778, -1.3716311, -3.6529503, -1.3840256, -2.1621041, 2.1629193
5: -5.8980103, -3.8909342, -5.9373207, -3.8688745, -1.9353900, 1.9515576
6: -16.8666611, -13.8310661, -16.8748188, -13.8383732, -2.2907674, 2.2306905
7: -4.6428032, -2.3045583, -4.6267056, -2.2980978, -2.1849670, 2.0884602
8: -5.2078705, -2.9642053, -5.2084432, -2.9504681, -1.6396139, 1.6423352
9: 4.4368038, 5.9540682, 4.4317560, 5.9636717, -1.3083980, 1.3057001

Time for backsubstitution: 14.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9208639, upper bound: 0.9181092
time: 3.72 seconds

## Relational analysis of IS_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9208638, upper bound: 0.9181093
time: 3.58 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.5790396, -9.4956741, -12.6096153, -9.4818821, -2.4618096, 2.6370564
1: -11.6996326, -9.2247009, -11.7169218, -9.2003822, -2.0322642, 1.9811001
2: -8.0681782, -6.2320747, -8.1146955, -6.2204432, -1.6684928, 1.7219431
3: -7.6602097, -5.1931157, -7.6746378, -5.1576624, -2.1709356, 2.1187711
4: -3.6586297, -1.3686562, -3.6531122, -1.3835149, -2.1671557, 2.1661210
5: -5.9032788, -3.8845017, -5.9382939, -3.8687651, -1.9476790, 1.9600954
6: -16.8781319, -13.8188076, -16.8771210, -13.8382053, -2.2992382, 2.2450559
7: -4.6457314, -2.3009458, -4.6269355, -2.2978895, -2.1875734, 2.0921669
8: -5.2124214, -2.9584470, -5.2085476, -2.9496884, -1.6452060, 1.6488159
9: 4.4332881, 5.9559450, 4.4315367, 5.9640255, -1.3124027, 1.3092791

Time for backsubstitution: 14.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5799
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9208638, upper bound: 0.9188830
time: 3.57 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9208639, upper bound: 0.9188831
time: 3.61 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -12.5695992, -9.5244799, -12.6202183, -9.4634056, -2.4732866, 2.6187444
1: -11.6908426, -9.2366714, -11.7295809, -9.1882257, -2.0312672, 1.9841945
2: -8.0497160, -6.2521429, -8.1374731, -6.2034431, -1.6699357, 1.7233729
3: -7.6269441, -5.2209206, -7.7179842, -5.1391625, -2.1605158, 2.1311941
4: -3.6373177, -1.3955841, -3.6731915, -1.3468938, -2.1847334, 2.1572628
5: -5.8948550, -3.9073162, -5.9478450, -3.8447824, -1.9551311, 1.9545207
6: -16.8591957, -13.8542118, -16.8981056, -13.8041973, -2.3145437, 2.2318404
7: -4.6072283, -2.3321939, -4.6789947, -2.2635541, -2.1803370, 2.1173105
8: -5.1929550, -2.9719386, -5.2298040, -2.9345465, -1.6436102, 1.6516846
9: 4.4472542, 5.9525614, 4.4123626, 5.9677410, -1.3056693, 1.3207314

Time for backsubstitution: 13.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9208638, upper bound: 0.9205365
time: 3.74 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9208639, upper bound: 0.9213105
time: 3.61 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.5779982, -9.5066538, -12.6202183, -9.4634056, -2.4795017, 2.6387072
1: -11.7001743, -9.2250233, -11.7295809, -9.1882257, -2.0511484, 2.0026698
2: -8.0719080, -6.2380018, -8.1374731, -6.2034431, -1.6929030, 1.7402437
3: -7.6701522, -5.2049541, -7.7179842, -5.1391625, -2.1854515, 2.1326582
4: -3.6570032, -1.3592918, -3.6731915, -1.3468938, -2.1878734, 2.1786771
5: -5.9032845, -3.8842745, -5.9478450, -3.8447824, -1.9668212, 1.9828629
6: -16.8792229, -13.8215942, -16.8981056, -13.8041973, -2.3224249, 2.2484341
7: -4.6579862, -2.2996984, -4.6789947, -2.2635541, -2.2013693, 2.1166725
8: -5.2140436, -2.9587955, -5.2298040, -2.9345465, -1.6554923, 1.6570587
9: 4.4324603, 5.9562154, 4.4123626, 5.9677410, -1.3266134, 1.3316751

Time for backsubstitution: 13.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9208639, upper bound: 0.9202422
time: 3.89 seconds

## Relational analysis of IS_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9208639, upper bound: 0.9210160
time: 3.71 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -12.6155739, -9.4646454, -12.5763569, -9.5111628, -2.6262217, 2.4812112
1: -11.7230082, -9.1898232, -11.6982298, -9.2280436, -1.9844608, 2.0400796
2: -8.1397610, -6.2167883, -8.0672626, -6.2420788, -1.7358475, 1.6730227
3: -7.6798916, -5.1334405, -7.6583548, -5.2068830, -2.0985150, 2.1763165
4: -3.6571112, -1.3792863, -3.6548641, -1.3695045, -2.1620698, 2.1617262
5: -5.9448314, -3.8526349, -5.9018660, -3.8905776, -1.9662395, 1.9595432
6: -16.8820229, -13.8317642, -16.8762646, -13.8306608, -2.2367611, 2.3085797
7: -4.6346989, -2.2920003, -4.6436148, -2.3040099, -2.0983071, 2.1834002
8: -5.2104793, -2.9404683, -5.2082896, -2.9611993, -1.6458116, 1.6504521
9: 4.4248171, 5.9677439, 4.4363756, 5.9555745, -1.3143954, 1.3145888

Time for backsubstitution: 14.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 5799
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9181059, upper bound: 0.9232984
time: 3.29 seconds

## Relational analysis of IS_A2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9188798, upper bound: 0.9233004
time: 3.34 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -12.6155739, -9.4646454, -12.6238470, -9.4507685, -2.6624222, 2.6568849
1: -11.7230082, -9.1898232, -11.7326813, -9.1806765, -2.0226331, 2.0252168
2: -8.1397610, -6.2167883, -8.1576061, -6.2045031, -1.7328527, 1.7372105
3: -7.6798916, -5.1334405, -7.7114992, -5.1173677, -2.1794672, 2.1946969
4: -3.6571112, -1.3792863, -3.6749346, -1.3529272, -2.2098784, 2.1996241
5: -5.9448314, -3.8526349, -5.9526811, -3.8351758, -1.9976945, 1.9969747
6: -16.8820229, -13.8317642, -16.8997612, -13.8071671, -2.3667369, 2.3629210
7: -4.6346989, -2.2920003, -4.6721177, -2.2623644, -2.2323680, 2.2417345
8: -5.2104793, -2.9404683, -5.2260323, -2.9283113, -1.6709235, 1.6690331
9: 4.4248171, 5.9677439, 4.4106207, 5.9707699, -1.3332663, 1.3392565

Time for backsubstitution: 14.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9188824, upper bound: 0.9208667
time: 3.62 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9188824, upper bound: 0.9233007
time: 3.87 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -12.6260729, -9.4461412, -12.5779991, -9.5066519, -2.6463399, 2.4979610
1: -11.7355156, -9.1776333, -11.7001781, -9.2250204, -2.0077724, 2.0547929
2: -8.1623917, -6.1997986, -8.0719109, -6.2380009, -1.7636814, 1.6945565
3: -7.7232900, -5.1149921, -7.6701622, -5.2049522, -2.1260211, 2.2070429
4: -3.6771855, -1.3426216, -3.6570044, -1.3592873, -2.1906786, 2.1862967
5: -5.9543195, -3.8286245, -5.9032869, -3.8842707, -1.9872231, 1.9828234
6: -16.9029350, -13.7977257, -16.8792248, -13.8215885, -2.2674022, 2.3308592
7: -4.6868372, -2.2577446, -4.6579971, -2.2996969, -2.1256480, 2.2277977
8: -5.2317557, -2.9253788, -5.2140493, -2.9587946, -1.6567924, 1.6714540
9: 4.4055681, 5.9714327, 4.4324579, 5.9562163, -1.3373656, 1.3256304

Time for backsubstitution: 14.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5799
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9213098, upper bound: 0.9208689
time: 3.63 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9213099, upper bound: 0.9208665
time: 3.45 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -12.6260729, -9.4461412, -12.6260767, -9.4461374, -2.6826553, 2.6795688
1: -11.7355156, -9.1776333, -11.7355165, -9.1776304, -2.0460815, 2.0408030
2: -8.1623917, -6.1997986, -8.1623936, -6.1997938, -1.7605543, 1.7595732
3: -7.7232900, -5.1149921, -7.7232981, -5.1149902, -2.2073088, 2.2244971
4: -3.6771855, -1.3426216, -3.6771874, -1.3426166, -2.2384377, 2.2232327
5: -5.9543195, -3.8286245, -5.9543214, -3.8286192, -2.0188899, 2.0228591
6: -16.9029350, -13.7977257, -16.9029369, -13.7977180, -2.4005117, 2.3855975
7: -4.6868372, -2.2577446, -4.6868491, -2.2577415, -2.2586570, 2.2877016
8: -5.2317557, -2.9253788, -5.2317619, -2.9253788, -1.6833539, 1.6900251
9: 4.4055681, 5.9714327, 4.4055643, 5.9714341, -1.3563225, 1.3512630

Time for backsubstitution: 14.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 5790
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9213098, upper bound: 0.9227796
time: 3.53 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9213100, upper bound: 0.9248072
time: 3.53 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 21.31 seconds
IS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 9, lower bound: -0.9208639, upper bound: 0.9181092
IS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 9, lower bound: -0.9208638, upper bound: 0.9181093
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 9, lower bound: -0.9208638, upper bound: 0.9188830
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 9, lower bound: -0.9208639, upper bound: 0.9188831
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 9, lower bound: -0.9208638, upper bound: 0.9205365
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 9, lower bound: -0.9208639, upper bound: 0.9213105
IS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 9, lower bound: -0.9208639, upper bound: 0.9202422
IS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 9, lower bound: -0.9208639, upper bound: 0.9210160
IS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 9, lower bound: -0.9181059, upper bound: 0.9232984
IS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 9, lower bound: -0.9188798, upper bound: 0.9233004
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 9, lower bound: -0.9188824, upper bound: 0.9208667
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 9, lower bound: -0.9188824, upper bound: 0.9233007
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 9, lower bound: -0.9213098, upper bound: 0.9208689
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 9, lower bound: -0.9213099, upper bound: 0.9208665
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 9, lower bound: -0.9213098, upper bound: 0.9227796
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 21.31
Output dim: 9, lower bound: -0.9213100, upper bound: 0.9248072

## BFS IS instance: IS_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -12.5557985, -9.5246735, -12.6062183, -9.4819422, -2.4413557, 2.6008480
1: -11.6890669, -9.2372208, -11.7163677, -9.2005081, -2.0156937, 1.9684379
2: -8.0416355, -6.2525535, -8.1127548, -6.2206130, -1.6440043, 1.6987758
3: -7.6172152, -5.2212663, -7.6723385, -5.1578236, -2.1319928, 2.0883563
4: -3.6367331, -1.3977122, -3.6529503, -1.3840256, -2.1457043, 2.1365941
5: -5.8909998, -3.9076686, -5.9373207, -3.8688745, -1.9227405, 1.9363613
6: -16.8495903, -13.8546124, -16.8748188, -13.8383732, -2.2716877, 2.2073331
7: -4.6064076, -2.3327527, -4.6267056, -2.2980978, -2.1484795, 2.0634546
8: -5.1925311, -2.9749517, -5.2084432, -2.9504681, -1.6270235, 1.6293448
9: 4.4476795, 5.9510565, 4.4317560, 5.9636717, -1.2981098, 1.2983091

Time for backsubstitution: 13.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4628

## Relational analysis of IS_A1_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9183878, upper bound: 0.9181042
time: 3.41 seconds

## Relational analysis of IS_A1_B1_A1_A1_A2

### Relational analysis result of IS_A1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9208589, upper bound: 0.9181043
time: 3.53 seconds

## BFS IS instance: IS_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -12.5642014, -9.5068512, -12.6062183, -9.4819422, -2.4515457, 2.6239748
1: -11.6983986, -9.2255669, -11.7163677, -9.2005081, -2.0270481, 1.9803202
2: -8.0638199, -6.2384143, -8.1127548, -6.2206130, -1.6651220, 1.7144189
3: -7.6604395, -5.2053022, -7.6723385, -5.1578236, -2.1646960, 2.1044514
4: -3.6564171, -1.3614161, -3.6529503, -1.3840256, -2.1637936, 2.1732142
5: -5.8994303, -3.8846354, -5.9373207, -3.8688745, -1.9320550, 1.9605732
6: -16.8696213, -13.8219967, -16.8748188, -13.8383732, -2.2944863, 2.2396934
7: -4.6571789, -2.3002496, -4.6267056, -2.2980978, -2.1992221, 2.0913415
8: -5.2136230, -2.9617991, -5.2084432, -2.9504681, -1.6458046, 1.6422923
9: 4.4328890, 5.9547091, 4.4317560, 5.9636717, -1.3135872, 1.3041577

Time for backsubstitution: 13.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4628

## Relational analysis of IS_A1_B1_A1_A2_A1

### Relational analysis result of IS_A1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9183878, upper bound: 0.9181042
time: 3.68 seconds

## Relational analysis of IS_A1_B1_A1_A2_A2

### Relational analysis result of IS_A1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9208588, upper bound: 0.9181042
time: 3.57 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -12.5722809, -9.5089931, -12.6096153, -9.4818821, -2.4534678, 2.6199498
1: -11.6922483, -9.2333393, -11.7169218, -9.2003822, -2.0216751, 1.9733047
2: -8.0506458, -6.2421389, -8.1146955, -6.2204432, -1.6511045, 1.7109489
3: -7.6287971, -5.2071543, -7.6746378, -5.1576624, -2.1408939, 2.1046591
4: -3.6410840, -1.3947358, -3.6531122, -1.3835149, -2.1507525, 2.1397886
5: -5.8962679, -3.9012523, -5.9382939, -3.8687651, -1.9350309, 1.9448721
6: -16.8610611, -13.8423557, -16.8771210, -13.8382053, -2.2801580, 2.2216969
7: -4.6093516, -2.3291690, -4.6269355, -2.2978895, -2.1510820, 2.0671523
8: -5.1970825, -2.9691768, -5.2085476, -2.9496884, -1.6326180, 1.6358172
9: 4.4441676, 5.9529352, 4.4315367, 5.9640255, -1.3021064, 1.3018876

Time for backsubstitution: 13.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4628

## Relational analysis of IS_A1_B1_A2_A1_A1

### Relational analysis result of IS_A1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9183878, upper bound: 0.9188780
time: 3.59 seconds

## Relational analysis of IS_A1_B1_A2_A1_A2

### Relational analysis result of IS_A1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9208588, upper bound: 0.9188781
time: 3.65 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -12.5806799, -9.4911633, -12.6096153, -9.4818821, -2.4636664, 2.6430831
1: -11.7015696, -9.2216749, -11.7169218, -9.2003822, -2.0330410, 1.9852037
2: -8.0728168, -6.2279987, -8.1146955, -6.2204432, -1.6722102, 1.7265911
3: -7.6720057, -5.1911883, -7.6746378, -5.1576624, -2.1736193, 2.1167250
4: -3.6607685, -1.3584409, -3.6531122, -1.3835149, -2.1688471, 2.1764164
5: -5.9046988, -3.8781958, -5.9382939, -3.8687651, -1.9443417, 1.9691148
6: -16.8810902, -13.8097363, -16.8771210, -13.8382053, -2.3029552, 2.2510850
7: -4.6601005, -2.2966342, -4.6269355, -2.2978895, -2.2018309, 2.0950511
8: -5.2181773, -2.9560452, -5.2085476, -2.9496884, -1.6514006, 1.6487709
9: 4.4293709, 5.9565854, 4.4315367, 5.9640255, -1.3175929, 1.3077379

Time for backsubstitution: 14.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 4628

## Relational analysis of IS_A1_B1_A2_A2_A1

### Relational analysis result of IS_A1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9183878, upper bound: 0.9188780
time: 3.35 seconds

## Relational analysis of IS_A1_B1_A2_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9208589, upper bound: 0.9188762
time: 3.63 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -12.5557985, -9.5246735, -12.6168079, -9.4634638, -2.4599538, 2.6150117
1: -11.6890669, -9.2372208, -11.7290268, -9.1883507, -2.0277557, 1.9810658
2: -8.0416355, -6.2525535, -8.1355247, -6.2036119, -1.6613493, 1.7210121
3: -7.6172152, -5.2212663, -7.7156820, -5.1393223, -2.1501923, 2.1270697
4: -3.6367331, -1.3977122, -3.6730294, -1.3474064, -2.1833925, 2.1547794
5: -5.8909998, -3.9076686, -5.9468679, -3.8448927, -1.9466867, 1.9484646
6: -16.8495903, -13.8546124, -16.8957958, -13.8043594, -2.3050172, 2.2293308
7: -4.6064076, -2.3327527, -4.6787658, -2.2637632, -2.1786480, 2.1163807
8: -5.1925311, -2.9749517, -5.2296972, -2.9353275, -1.6418915, 1.6479782
9: 4.4476795, 5.9510565, 4.4125834, 5.9673843, -1.3039958, 1.3178799

Time for backsubstitution: 14.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4628

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9208588, upper bound: 0.9180607
time: 4.05 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9208588, upper bound: 0.9205316
time: 3.61 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -12.5722809, -9.5089931, -12.6202030, -9.4634056, -2.4720664, 2.6341119
1: -11.6922483, -9.2333393, -11.7295790, -9.1882267, -2.0337343, 1.9859333
2: -8.0506458, -6.2421389, -8.1374655, -6.2034435, -1.6684513, 1.7331858
3: -7.6287971, -5.2071543, -7.7179775, -5.1391644, -2.1590915, 2.1323965
4: -3.6410840, -1.3947358, -3.6731906, -1.3468974, -2.1884408, 2.1579750
5: -5.8962679, -3.9012523, -5.9478421, -3.8447833, -1.9589758, 1.9569743
6: -16.8610611, -13.8423557, -16.8980980, -13.8041973, -2.3134880, 2.2436941
7: -4.6093516, -2.3291690, -4.6789927, -2.2635550, -2.1812515, 2.1196737
8: -5.1970825, -2.9691768, -5.2298040, -2.9345489, -1.6474853, 1.6544472
9: 4.4441676, 5.9529352, 4.4123650, 5.9677382, -1.3079937, 1.3214593

Time for backsubstitution: 13.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 6166
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4628

## Relational analysis of IS_A1_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9183878, upper bound: 0.9213055
time: 3.51 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2

### Relational analysis result of IS_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9208588, upper bound: 0.9213055
time: 3.58 seconds

## BFS IS instance: IS_A1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -12.5642014, -9.5068512, -12.6168079, -9.4634638, -2.4661741, 2.6349764
1: -11.6983986, -9.2255669, -11.7290268, -9.1883507, -2.0476313, 1.9995568
2: -8.0638199, -6.2384143, -8.1355247, -6.2036119, -1.6843190, 1.7378821
3: -7.6604395, -5.2053022, -7.7156820, -5.1393223, -2.1751485, 2.1296849
4: -3.6564171, -1.3614161, -3.6730294, -1.3474064, -2.1865311, 2.1761992
5: -5.8994303, -3.8846354, -5.9468679, -3.8448927, -1.9583735, 1.9768093
6: -16.8696213, -13.8219967, -16.8957958, -13.8043594, -2.3129039, 2.2459245
7: -4.6571789, -2.3002496, -4.6787658, -2.2637632, -2.1996870, 2.1157360
8: -5.2136230, -2.9617991, -5.2296972, -2.9353275, -1.6537623, 1.6533407
9: 4.4328890, 5.9547091, 4.4125834, 5.9673843, -1.3249435, 1.3288300

Time for backsubstitution: 14.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4628

## Relational analysis of IS_A1_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9221094, upper bound: 0.9202370
time: 3.64 seconds

## Relational analysis of IS_A1_B2_A2_A1_A2

### Relational analysis result of IS_A1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9245803, upper bound: 0.9202371
time: 3.73 seconds

## BFS IS instance: IS_A1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -12.5806799, -9.4911633, -12.6202030, -9.4634056, -2.4782958, 2.6540854
1: -11.7015696, -9.2216749, -11.7295790, -9.1882267, -2.0536227, 2.0044408
2: -8.0728168, -6.2279987, -8.1374655, -6.2034435, -1.6914082, 1.7500548
3: -7.6720057, -5.1911883, -7.7179775, -5.1391644, -2.1840477, 2.1459756
4: -3.6607685, -1.3584409, -3.6731906, -1.3468974, -2.1915841, 2.1794016
5: -5.9046988, -3.8781958, -5.9478421, -3.8447833, -1.9706602, 1.9853272
6: -16.8810902, -13.8097363, -16.8980980, -13.8041973, -2.3213735, 2.2602901
7: -4.6601005, -2.2966342, -4.6789927, -2.2635550, -2.2022953, 2.1194451
8: -5.2181773, -2.9560452, -5.2298040, -2.9345489, -1.6593647, 1.6598389
9: 4.4293709, 5.9565854, 4.4123650, 5.9677382, -1.3289416, 1.3324063

Time for backsubstitution: 14.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4628

## Relational analysis of IS_A1_B2_A2_A2_A1

### Relational analysis result of IS_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9221095, upper bound: 0.9210109
time: 3.59 seconds

## Relational analysis of IS_A1_B2_A2_A2_A2

### Relational analysis result of IS_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9245802, upper bound: 0.9210110
time: 3.45 seconds

## BFS IS instance: IS_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -12.6121521, -9.4647036, -12.5625601, -9.5113621, -2.6224918, 2.4678822
1: -11.7224512, -9.1899471, -11.6964588, -9.2285872, -1.9813342, 2.0365765
2: -8.1378164, -6.2169561, -8.0591784, -6.2424893, -1.7328362, 1.6644418
3: -7.6775846, -5.1335993, -7.6486378, -5.2072296, -2.0943873, 2.1659944
4: -3.6569471, -1.3797987, -3.6542778, -1.3716311, -2.1595874, 2.1603847
5: -5.9438529, -3.8527422, -5.8980103, -3.8909342, -1.9601860, 1.9506559
6: -16.8797150, -13.8319283, -16.8666611, -13.8310661, -2.2342551, 2.2990484
7: -4.6344676, -2.2922082, -4.6428032, -2.3045583, -2.0973725, 2.1817164
8: -5.2103710, -2.9412508, -5.2078705, -2.9642053, -1.6420970, 1.6487486
9: 4.4250383, 5.9673891, 4.4368038, 5.9540682, -1.3115447, 1.3129139

Time for backsubstitution: 13.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5799
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of IS_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9181066, upper bound: 0.9208639
time: 4.34 seconds

## Relational analysis of IS_A2_A1_B1_B1_B2

### Relational analysis result of IS_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9181066, upper bound: 0.9233004
time: 7.01 seconds

## BFS IS instance: IS_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -12.6155586, -9.4646463, -12.5790396, -9.4956741, -2.6347141, 2.4800065
1: -11.7230062, -9.1898241, -11.6996326, -9.2247009, -1.9862070, 2.0425518
2: -8.1397533, -6.2167883, -8.0681782, -6.2320747, -1.7372534, 1.6715331
3: -7.6798840, -5.1334419, -7.6602097, -5.1931157, -2.0997200, 2.1749253
4: -3.6571102, -1.3792887, -3.6586297, -1.3686562, -2.1627941, 2.1654344
5: -5.9448261, -3.8526349, -5.9032788, -3.8845017, -1.9687228, 1.9589027
6: -16.8820152, -13.8317661, -16.8781319, -13.8188076, -2.2486131, 2.3075290
7: -4.6346984, -2.2920012, -4.6457314, -2.3009458, -2.1010804, 2.1843235
8: -5.2104793, -2.9404716, -5.2124214, -2.9584470, -1.6485782, 1.6543400
9: 4.4248180, 5.9677429, 4.4332881, 5.9559450, -1.3151228, 1.3169179

Time for backsubstitution: 14.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5799
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of IS_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9188804, upper bound: 0.9208642
time: 3.80 seconds

## Relational analysis of IS_A2_A1_B1_B2_B2

### Relational analysis result of IS_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9188805, upper bound: 0.9233005
time: 3.54 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -12.6155739, -9.4646454, -12.6155739, -9.4646454, -2.6446066, 2.6446066
1: -11.7230082, -9.1898232, -11.7230082, -9.1898232, -2.0142760, 2.0142760
2: -8.1397610, -6.2167883, -8.1397610, -6.2167883, -1.7198133, 1.7198136
3: -7.6798916, -5.1334405, -7.6798916, -5.1334405, -2.1635871, 2.1635871
4: -3.6571112, -1.3792863, -3.6571112, -1.3792863, -2.1831579, 2.1831582
5: -5.9448314, -3.8526349, -5.9448314, -3.8526349, -1.9817495, 1.9817498
6: -16.8820229, -13.8317642, -16.8820229, -13.8317642, -2.3426328, 2.3426328
7: -4.6346989, -2.2920003, -4.6346989, -2.2920003, -2.2053461, 2.2053459
8: -5.2104793, -2.9404683, -5.2104793, -2.9404683, -1.6563377, 1.6563377
9: 4.4248171, 5.9677439, 4.4248171, 5.9677439, -1.3258445, 1.3258446

Time for backsubstitution: 13.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 5790
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A2_A1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9223268, upper bound: 0.9219882
time: 5.81 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9223267, upper bound: 0.9227747
time: 4.13 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -12.6155739, -9.4646454, -12.6260729, -9.4461412, -2.6685624, 2.6586945
1: -11.7230082, -9.1898232, -11.7355156, -9.1776333, -2.0268483, 2.0268970
2: -8.1397610, -6.2167883, -8.1623917, -6.1997986, -1.7380981, 1.7410393
3: -7.6798916, -5.1334405, -7.7232900, -5.1149921, -2.1817088, 2.2063696
4: -3.6571112, -1.3792863, -3.6771855, -1.3426216, -2.2202740, 2.2013171
5: -5.9448314, -3.8526349, -5.9543195, -3.8286245, -2.0069189, 1.9937153
6: -16.8820229, -13.8317642, -16.9029350, -13.7977257, -2.3759804, 2.3671579
7: -4.6346989, -2.2920003, -4.6868372, -2.2577446, -2.2347984, 2.2434862
8: -5.2104793, -2.9404683, -5.2317557, -2.9253788, -1.6711435, 1.6752152
9: 4.4248171, 5.9677439, 4.4055681, 5.9714327, -1.3316972, 1.3454064

Time for backsubstitution: 14.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 5790
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 5860

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9215406, upper bound: 0.9251428
time: 5.93 seconds

## Relational analysis of IS_A2_A1_B2_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9223268, upper bound: 0.9251454
time: 4.48 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -12.6260729, -9.4461412, -12.5695992, -9.5244799, -2.6232104, 2.4888780
1: -11.7355156, -9.1776333, -11.6908426, -9.2366714, -1.9892936, 2.0417156
2: -8.1623917, -6.1997986, -8.0497160, -6.2521429, -1.7468081, 1.6730654
3: -7.7232900, -5.1149921, -7.6269441, -5.2209206, -2.1082625, 2.1652234
4: -3.6771855, -1.3426216, -3.6373177, -1.3955841, -2.1540580, 2.1831546
5: -5.9543195, -3.8286245, -5.8948550, -3.9073162, -1.9630117, 1.9638425
6: -16.9029350, -13.7977257, -16.8591957, -13.8542118, -2.2350366, 2.3229747
7: -4.6868372, -2.2577446, -4.6072283, -2.3321939, -2.1238787, 2.1770530
8: -5.2317557, -2.9253788, -5.1929550, -2.9719386, -1.6514888, 1.6526823
9: 4.4055681, 5.9714327, 4.4472542, 5.9525614, -1.3265653, 1.3101475

Time for backsubstitution: 14.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9205340, upper bound: 0.9208663
time: 3.26 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9213078, upper bound: 0.9208663
time: 3.77 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -12.6260729, -9.4461412, -12.5779982, -9.5066538, -2.6432490, 2.4976470
1: -11.7355156, -9.1776333, -11.7001743, -9.2250233, -2.0077686, 2.0615966
2: -8.1623917, -6.1997986, -8.0719080, -6.2380018, -1.7636783, 1.6960320
3: -7.7232900, -5.1149921, -7.6701522, -5.2049541, -2.1238854, 2.1961682
4: -3.6771855, -1.3426216, -3.6570032, -1.3592918, -2.1754723, 2.1862953
5: -5.9543195, -3.8286245, -5.9032845, -3.8842745, -1.9914589, 1.9828186
6: -16.9029350, -13.7977257, -16.8792229, -13.8215942, -2.2516305, 2.3308563
7: -4.6868372, -2.2577446, -4.6579862, -2.2996984, -2.1256464, 2.1980844
8: -5.2317557, -2.9253788, -5.2140436, -2.9587955, -1.6567888, 1.6647320
9: 4.4055681, 5.9714327, 4.4324603, 5.9562154, -1.3373637, 1.3311523

Time for backsubstitution: 13.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9205342, upper bound: 0.9230058
time: 3.64 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9213080, upper bound: 0.9208664
time: 3.37 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -12.6260729, -9.4461412, -12.6155739, -9.4646454, -2.6586943, 2.6685624
1: -11.7355156, -9.1776333, -11.7230082, -9.1898232, -2.0268970, 2.0268488
2: -8.1623917, -6.1997986, -8.1397610, -6.2167883, -1.7410388, 1.7380984
3: -7.7232900, -5.1149921, -7.6798916, -5.1334405, -2.2063699, 2.1817081
4: -3.6771855, -1.3426216, -3.6571112, -1.3792863, -2.2013168, 2.2202742
5: -5.9543195, -3.8286245, -5.9448314, -3.8526349, -1.9937153, 2.0069189
6: -16.9029350, -13.7977257, -16.8820229, -13.8317642, -2.3671579, 2.3759804
7: -4.6868372, -2.2577446, -4.6346989, -2.2920003, -2.2434864, 2.2347987
8: -5.2317557, -2.9253788, -5.2104793, -2.9404683, -1.6752157, 1.6711433
9: 4.4055681, 5.9714327, 4.4248171, 5.9677439, -1.3454065, 1.3316973

Time for backsubstitution: 14.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 5790
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 5860

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9246722, upper bound: 0.9219883
time: 4.28 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9246722, upper bound: 0.9227741
time: 4.63 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -12.6260729, -9.4461412, -12.6260729, -9.4461412, -2.6795654, 2.6795654
1: -11.7355156, -9.1776333, -11.7355156, -9.1776333, -2.0460787, 2.0460787
2: -8.1623917, -6.1997986, -8.1623917, -6.1997986, -1.7605505, 1.7605500
3: -7.7232900, -5.1149921, -7.7232900, -5.1149921, -2.2073073, 2.2073069
4: -3.6771855, -1.3426216, -3.6771855, -1.3426216, -2.2232313, 2.2232316
5: -5.9543195, -3.8286245, -5.9543195, -3.8286245, -2.0228543, 2.0228541
6: -16.9029350, -13.7977257, -16.9029350, -13.7977257, -2.3855946, 2.3855946
7: -4.6868372, -2.2577446, -4.6868372, -2.2577446, -2.2586560, 2.2586558
8: -5.2317557, -2.9253788, -5.2317557, -2.9253788, -1.6833501, 1.6833508
9: 4.4055681, 5.9714327, 4.4055681, 5.9714327, -1.3563206, 1.3563204

Time for backsubstitution: 14.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 5860

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4670

## Relational analysis of IS_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9246723, upper bound: 0.9219881
time: 4.37 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9246723, upper bound: 0.9227745
time: 6.79 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 25.75 seconds
IS_A1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9183878, upper bound: 0.9181042
IS_A1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9208589, upper bound: 0.9181043
IS_A1_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9183878, upper bound: 0.9181042
IS_A1_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9208588, upper bound: 0.9181042
IS_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9183878, upper bound: 0.9188780
IS_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9208588, upper bound: 0.9188781
IS_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9183878, upper bound: 0.9188780
IS_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9208589, upper bound: 0.9188762
IS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9208588, upper bound: 0.9180607
IS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9208588, upper bound: 0.9205316
IS_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9183878, upper bound: 0.9213055
IS_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9208588, upper bound: 0.9213055
IS_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9221094, upper bound: 0.9202370
IS_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9245803, upper bound: 0.9202371
IS_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9221095, upper bound: 0.9210109
IS_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9245802, upper bound: 0.9210110
IS_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9181066, upper bound: 0.9208639
IS_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9181066, upper bound: 0.9233004
IS_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9188804, upper bound: 0.9208642
IS_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9188805, upper bound: 0.9233005
IS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9223268, upper bound: 0.9219882
IS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9223267, upper bound: 0.9227747
IS_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9215406, upper bound: 0.9251428
IS_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9223268, upper bound: 0.9251454
IS_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9205340, upper bound: 0.9208663
IS_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9213078, upper bound: 0.9208663
IS_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9205342, upper bound: 0.9230058
IS_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9213080, upper bound: 0.9208664
IS_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9246722, upper bound: 0.9219883
IS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9246722, upper bound: 0.9227741
IS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9246723, upper bound: 0.9219881
IS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.75
Output dim: 9, lower bound: -0.9246723, upper bound: 0.9227745

## BFS IS instance: IS_A1_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -12.5444050, -9.5432253, -12.6054649, -9.4867620, -2.4308434, 2.5800843
1: -11.6838140, -9.2427711, -11.7155819, -9.2018318, -2.0082359, 1.9612708
2: -8.0355082, -6.2547770, -8.1119967, -6.2215075, -1.6345181, 1.6528571
3: -7.5977836, -5.2287922, -7.6674609, -5.1583128, -2.1132350, 2.0754023
4: -3.6302321, -1.4009244, -3.6516218, -1.3844998, -2.1383142, 2.1494627
5: -5.8826132, -3.9123535, -5.9352770, -3.8693225, -1.9118271, 1.9270239
6: -16.8407898, -13.8642454, -16.8741932, -13.8409729, -2.2582905, 2.1972659
7: -4.6010265, -2.3440068, -4.6261492, -2.3008299, -2.1399593, 2.0613017
8: -5.1857648, -2.9945154, -5.2075901, -2.9554572, -1.5866246, 1.6077882
9: 4.4572821, 5.9468098, 4.4341736, 5.9631972, -1.2884717, 1.2902967

Time for backsubstitution: 14.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 6166
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A1_B1_A1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9176143, upper bound: 0.9181043
time: 3.64 seconds

## Relational analysis of IS_A1_B1_A1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9176143, upper bound: 0.9181025
time: 3.98 seconds

## BFS IS instance: IS_A1_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -12.5557966, -9.5246878, -12.6062183, -9.4819422, -2.4411583, 2.5917673
1: -11.6890650, -9.2372236, -11.7163677, -9.2005081, -2.0156932, 1.9659579
2: -8.0416365, -6.2525544, -8.1127548, -6.2206130, -1.6417060, 1.7085841
3: -7.6172056, -5.2212672, -7.6723385, -5.1578236, -2.1177039, 2.0883563
4: -3.6367311, -1.3977129, -3.6529503, -1.3840256, -2.1457934, 2.1365921
5: -5.8909960, -3.9076700, -5.9373207, -3.8688745, -1.9227338, 1.9414945
6: -16.8495884, -13.8546162, -16.8748188, -13.8383732, -2.2706697, 2.2073865
7: -4.6064086, -2.3327589, -4.6267056, -2.2980978, -2.1484776, 2.0583184
8: -5.1925297, -2.9749594, -5.2084432, -2.9504681, -1.6270225, 1.6249260
9: 4.4476833, 5.9510565, 4.4317560, 5.9636717, -1.2930913, 1.2983079

Time for backsubstitution: 14.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A1_B1_A1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9200854, upper bound: 0.9181043
time: 3.89 seconds

## Relational analysis of IS_A1_B1_A1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9200854, upper bound: 0.9181043
time: 3.80 seconds

## BFS IS instance: IS_A1_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: -12.5528069, -9.5253763, -12.6054649, -9.4867620, -2.4409585, 2.6032352
1: -11.6931362, -9.2310715, -11.7155819, -9.2018318, -2.0195675, 1.9731998
2: -8.0576267, -6.2406335, -8.1119967, -6.2215075, -1.6556201, 1.6684992
3: -7.6410584, -5.2128248, -7.6674609, -5.1583128, -2.1458123, 2.0915017
4: -3.6498780, -1.3646340, -3.6516218, -1.3844998, -2.1563630, 2.1860671
5: -5.8910465, -3.8893399, -5.9352770, -3.8693225, -1.9211893, 1.9512138
6: -16.8607979, -13.8316269, -16.8741932, -13.8409729, -2.2810614, 2.2296274
7: -4.6517897, -2.3114676, -4.6261492, -2.3008299, -2.1906853, 2.0892355
8: -5.2068667, -2.9813433, -5.2075901, -2.9554572, -1.6054220, 1.6208031
9: 4.4424810, 5.9504385, 4.4341736, 5.9631972, -1.3039539, 1.2961270

Time for backsubstitution: 14.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 6166
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 5790

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A1_B1_A1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9200535, upper bound: 0.9181042
time: 3.58 seconds

## Relational analysis of IS_A1_B1_A1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9200535, upper bound: 0.9181043
time: 3.59 seconds

## BFS IS instance: IS_A1_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: -12.5642014, -9.5068655, -12.6062183, -9.4819422, -2.4513502, 2.6149313
1: -11.6983976, -9.2255697, -11.7163677, -9.2005081, -2.0270472, 1.9778578
2: -8.0638199, -6.2384148, -8.1127548, -6.2206130, -1.6628308, 1.7242274
3: -7.6604304, -5.2053022, -7.6723385, -5.1578236, -2.1503861, 2.1044505
4: -3.6564145, -1.3614182, -3.6529503, -1.3840256, -2.1638842, 2.1732132
5: -5.8994265, -3.8846359, -5.9373207, -3.8688745, -1.9320478, 1.9657071
6: -16.8696194, -13.8220005, -16.8748188, -13.8383732, -2.2934735, 2.2397480
7: -4.6571784, -2.3002570, -4.6267056, -2.2980978, -2.1992216, 2.0862052
8: -5.2136207, -2.9618073, -5.2084432, -2.9504681, -1.6458027, 1.6378735
9: 4.4328918, 5.9547091, 4.4317560, 5.9636717, -1.3085804, 1.3041563

Time for backsubstitution: 14.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: B, layer: 1, pos: 5860
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 6166
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_A1_B1_A1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9225243, upper bound: 0.9181042
time: 3.81 seconds

## Relational analysis of IS_A1_B1_A1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.9225243, upper bound: 0.9181043
time: 3.82 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 22.29 seconds
IS_A1_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 22.29
Output dim: 9, lower bound: -0.9176143, upper bound: 0.9181043
IS_A1_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 22.29
Output dim: 9, lower bound: -0.9176143, upper bound: 0.9181025
IS_A1_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 22.29
Output dim: 9, lower bound: -0.9200854, upper bound: 0.9181043
IS_A1_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 22.29
Output dim: 9, lower bound: -0.9200854, upper bound: 0.9181043
IS_A1_B1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 22.29
Output dim: 9, lower bound: -0.9200535, upper bound: 0.9181042
IS_A1_B1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 22.29
Output dim: 9, lower bound: -0.9200535, upper bound: 0.9181043
IS_A1_B1_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 22.29
Output dim: 9, lower bound: -0.9225243, upper bound: 0.9181042
IS_A1_B1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 22.29
Output dim: 9, lower bound: -0.9225243, upper bound: 0.9181043
IS_A1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9183878, upper bound: 0.9188780
IS_A1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9208588, upper bound: 0.9188781
IS_A1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9183878, upper bound: 0.9188780
IS_A1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9208589, upper bound: 0.9188762
IS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9208588, upper bound: 0.9180607
IS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9208588, upper bound: 0.9205316
IS_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9183878, upper bound: 0.9213055
IS_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9208588, upper bound: 0.9213055
IS_A1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9221094, upper bound: 0.9202370
IS_A1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9245803, upper bound: 0.9202371
IS_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9221095, upper bound: 0.9210109
IS_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9245802, upper bound: 0.9210110
IS_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9181066, upper bound: 0.9208639
IS_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9181066, upper bound: 0.9233004
IS_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9188804, upper bound: 0.9208642
IS_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9188805, upper bound: 0.9233005
IS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9223268, upper bound: 0.9219882
IS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9223267, upper bound: 0.9227747
IS_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9215406, upper bound: 0.9251428
IS_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9223268, upper bound: 0.9251454
IS_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9205340, upper bound: 0.9208663
IS_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9213078, upper bound: 0.9208663
IS_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9205342, upper bound: 0.9230058
IS_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9213080, upper bound: 0.9208664
IS_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9246722, upper bound: 0.9219883
IS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9246722, upper bound: 0.9227741
IS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9246723, upper bound: 0.9219881
IS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 22.29
Output dim: 9, lower bound: -0.9246723, upper bound: 0.9227745
Binary search (step 1): status=Status.UNKNOWN, k_low=6, k_high=8, k_mid=7, eps_mid=0.0273438, abs_max=1.353825569152832
rel_dist={9: [-0.9264248152800034, 0.9264272454017544]}

## Binary search (step 2) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6166
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 5799
type: B, layer: 1, pos: 5799
type: A, layer: 1, pos: 4670
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 885
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 5860

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 6166

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8618511, upper bound: 0.8638822
time: 6.09 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8660041, upper bound: 0.8660042
time: 5.88 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.18 seconds
IS_B1, status: Status.VERIFIED, split count: 1, time: 12.18
Output dim: 9, lower bound: -0.8618511, upper bound: 0.8638822
IS_B2, status: Status.UNKNOWN, split count: 1, time: 12.18
Output dim: 9, lower bound: -0.8660041, upper bound: 0.8660042

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -12.6260805, -9.4460983, -12.6260767, -9.4461374, -2.5581980, 2.5851147
1: -11.7355251, -9.1776180, -11.7355165, -9.1776304, -1.9601488, 1.9660220
2: -8.1624165, -6.1997890, -8.1623936, -6.1997938, -1.7288234, 1.6838202
3: -7.7233047, -5.1149755, -7.7232981, -5.1149902, -2.1374745, 2.1893306
4: -3.6771922, -1.3426111, -3.6771874, -1.3426166, -2.1486778, 2.1735203
5: -5.9543285, -3.8286102, -5.9543214, -3.8286192, -1.9661827, 1.9913812
6: -16.9029427, -13.7977104, -16.9029369, -13.7977180, -2.2669287, 2.2457156
7: -4.6868625, -2.2577329, -4.6868491, -2.2577415, -2.2004547, 2.1784854
8: -5.2317648, -2.9253674, -5.2317619, -2.9253788, -1.6098168, 1.6115212
9: 4.4055543, 5.9714384, 4.4055643, 5.9714341, -1.3057528, 1.3060529

Time for backsubstitution: 14.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5799
type: A, layer: 1, pos: 5799
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 5860

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5799

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8648961, upper bound: 0.8627617
time: 7.75 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8660031, upper bound: 0.8660037
time: 4.28 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 26.57 seconds
IS_B2_B1, status: Status.VERIFIED, split count: 2, time: 26.57
Output dim: 9, lower bound: -0.8648961, upper bound: 0.8627617
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 26.57
Output dim: 9, lower bound: -0.8660031, upper bound: 0.8660037

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -12.6260815, -9.4460983, -12.6260729, -9.4461412, -2.5532942, 2.5836358
1: -11.7355242, -9.1776142, -11.7355156, -9.1776333, -1.9601083, 1.9710844
2: -8.1624165, -6.1997886, -8.1623917, -6.1997986, -1.7288122, 1.6847579
3: -7.7233057, -5.1149759, -7.7232900, -5.1149921, -2.1374722, 2.1710589
4: -3.6771927, -1.3426113, -3.6771855, -1.3426216, -2.1325760, 2.1735182
5: -5.9543290, -3.8286107, -5.9543195, -3.8286245, -1.9633946, 1.9857540
6: -16.9029408, -13.7977104, -16.9029350, -13.7977257, -2.2511370, 2.2457120
7: -4.6868610, -2.2577343, -4.6868372, -2.2577446, -2.1985016, 2.1470246
8: -5.2317662, -2.9253669, -5.2317557, -2.9253788, -1.6070638, 1.6012018
9: 4.4055548, 5.9714375, 4.4055681, 5.9714327, -1.3029134, 1.3077593

Time for backsubstitution: 14.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5799
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 5860
type: A, layer: 1, pos: 4628
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: A, layer: 1, pos: 6166
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 5860

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 5799

## Relational analysis of IS_B2_B2_A1

### Relational analysis result of IS_B2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8627620, upper bound: 0.8648987
time: 4.63 seconds

## Relational analysis of IS_B2_B2_A2

### Relational analysis result of IS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.8627620, upper bound: 0.8660059
time: 3.85 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 23.12 seconds
IS_B2_B2_A1, status: Status.VERIFIED, split count: 3, time: 23.12
Output dim: 9, lower bound: -0.8627620, upper bound: 0.8648987
IS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 23.12
Output dim: 9, lower bound: -0.8627620, upper bound: 0.8660059

## BFS IS instance: IS_B2_B2_A2

### Backsubstitution after applying IS history:
0: -12.6260805, -9.4461031, -12.6260729, -9.4461412, -2.5532908, 2.5802062
1: -11.7355223, -9.1776190, -11.7355156, -9.1776333, -1.9652081, 1.9710813
2: -8.1624136, -6.1997924, -8.1623917, -6.1997986, -1.7279260, 1.6847539
3: -7.7232966, -5.1149769, -7.7232900, -5.1149921, -2.1192732, 2.1692140
4: -3.6771913, -1.3426151, -3.6771855, -1.3426216, -2.1325750, 2.1574171
5: -5.9543266, -3.8286142, -5.9543195, -3.8286245, -1.9633889, 1.9885874
6: -16.9029388, -13.7977180, -16.9029350, -13.7977257, -2.2511346, 2.2299211
7: -4.6868505, -2.2577374, -4.6868372, -2.2577446, -2.1689916, 2.1470225
8: -5.2317591, -2.9253707, -5.2317557, -2.9253788, -1.5994935, 1.6011983
9: 4.4055586, 5.9714375, 4.4055681, 5.9714327, -1.3074579, 1.3077579

Time for backsubstitution: 14.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4670
type: A, layer: 1, pos: 4670
type: A, layer: 1, pos: 4628
type: A, layer: 1, pos: 6166
type: A, layer: 1, pos: 5860
type: B, layer: 1, pos: 4628
type: A, layer: 1, pos: 885
type: B, layer: 1, pos: 885
type: B, layer: 1, pos: 5790
type: A, layer: 1, pos: 5790
type: B, layer: 1, pos: 5860

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 4670

## Relational analysis of IS_B2_B2_A2_B1

### Relational analysis result of IS_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8620777, upper bound: 0.8627623
time: 3.96 seconds

## Relational analysis of IS_B2_B2_A2_B2

### Relational analysis result of IS_B2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.8627597, upper bound: 0.8641718
time: 4.16 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 22.60 seconds
IS_B2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 22.60
Output dim: 9, lower bound: -0.8620777, upper bound: 0.8627623
IS_B2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 22.60
Output dim: 9, lower bound: -0.8627597, upper bound: 0.8641718
Binary search (step 2): status=Status.VERIFIED, k_low=6, k_high=6, k_mid=6, eps_mid=0.0234375, abs_max=1.3057825565338135
rel_dist={9: [-0.8660068367423657, 0.8660078437084611]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0234375
execution time: 1749.58 seconds
