## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.1665576092
Search space: {k/256 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.0259066, -5.5622492, -9.0259066, -5.5622492, -3.4636574, 3.4636574)
1: (-6.5765400, -3.9590962, -6.5765400, -3.9590962, -2.6174438, 2.6174438)
2: (8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.6077518, 2.6077518)
3: (-6.1232843, -2.8826065, -6.1232843, -2.8826065, -3.2406778, 3.2406778)
4: (-11.8333845, -7.9824409, -11.8333845, -7.9824409, -3.8509436, 3.8509436)
5: (-13.6636562, -10.1825514, -13.6636562, -10.1825514, -3.4811049, 3.4811049)
6: (-15.6556635, -12.3171921, -15.6556635, -12.3171921, -3.2803464, 3.2803464)
7: (-5.5686188, -2.0476649, -5.5686188, -2.0476649, -3.5209539, 3.5209539)
8: (-1.9611964, 0.3840871, -1.9611964, 0.3840871, -2.3452835, 2.3452835)
9: (-7.3109250, -4.0054374, -7.3109250, -4.0054374, -3.3054876, 3.3054876)

## BASE Result
execution time: IAR + LP analysis = 14.65 + 33.79 = 48.45 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3551.55 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.5009913444519043
rel_dist={2: [-1.4846913114101667, 1.4846931118005955]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.2642369270324707
rel_dist={2: [-1.168894797061638, 1.1688945587998152]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.106400489807129
rel_dist={2: [-0.9214908145334242, 0.9214909870646206]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=2.1853187084198
rel_dist={2: [-1.048994002949069, 1.0489933180675663]}

## Binary Search Result
Binary search time: 221.56 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual) starts
Time budget: 3329.99 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4639
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 6191
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5704881, upper bound: 1.5533021
time: 5.14 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5739634, upper bound: 1.5739622
time: 5.08 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.47 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.47
Output dim: 2, lower bound: -1.5704881, upper bound: 1.5533021
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.47
Output dim: 2, lower bound: -1.5739634, upper bound: 1.5739622

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -8.9986153, -5.5798907, -9.0187588, -5.5666394, -3.0751257, 3.0720921
1: -6.5597734, -3.9739642, -6.5736613, -3.9632578, -2.4620223, 2.4716249
2: 8.3694963, 10.8838148, 8.3301010, 10.9183426, -2.5270705, 2.5256422
3: -6.0988617, -2.9099305, -6.1171312, -2.8889596, -3.2099020, 3.2072008
4: -11.8087769, -8.0043459, -11.8273172, -7.9849596, -3.3877220, 3.3890171
5: -13.6352262, -10.1946335, -13.6560574, -10.1831951, -2.9288874, 2.9359717
6: -15.6257658, -12.3342810, -15.6474390, -12.3202085, -2.7208977, 2.7126360
7: -5.5424242, -2.0679922, -5.5636425, -2.0532918, -3.4338775, 3.4393601
8: -1.9461651, 0.3789444, -1.9574537, 0.3831120, -2.2824059, 2.2862601
9: -7.2891226, -4.0242567, -7.3049994, -4.0082788, -2.9811802, 2.9796991

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6191
type: A, layer: 1, pos: 6191
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6191

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5694686, upper bound: 1.5533014
time: 9.14 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5704877, upper bound: 1.5533017
time: 5.46 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.0258865, -5.5622592, -9.0259018, -5.5622482, -3.1334143, 3.0955796
1: -6.5765324, -3.9591095, -6.5765390, -3.9590983, -2.4903359, 2.4968400
2: 8.3243237, 10.9320021, 8.3243122, 10.9320574, -2.5798974, 2.5709052
3: -6.1232662, -2.8826249, -6.1232829, -2.8826089, -3.2406573, 3.2406580
4: -11.8333654, -7.9824467, -11.8333836, -7.9824424, -3.4140549, 3.4182758
5: -13.6636353, -10.1825542, -13.6636515, -10.1825542, -2.9661427, 2.9807377
6: -15.6556358, -12.3172054, -15.6556606, -12.3171921, -2.7605534, 2.7380896
7: -5.5686011, -2.0476840, -5.5686188, -2.0476673, -3.4725504, 3.4703975
8: -1.9611835, 0.3840857, -1.9611955, 0.3840876, -2.3112822, 2.2939153
9: -7.3109016, -4.0054460, -7.3109245, -4.0054359, -2.9985704, 3.0072212

Time for backsubstitution: 14.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 6191
type: B, layer: 1, pos: 6191
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6191

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5739630, upper bound: 1.5729429
time: 4.73 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5739630, upper bound: 1.5739617
time: 5.22 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 35.00 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 35.00
Output dim: 2, lower bound: -1.5694686, upper bound: 1.5533014
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 35.00
Output dim: 2, lower bound: -1.5704877, upper bound: 1.5533017
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 35.00
Output dim: 2, lower bound: -1.5739630, upper bound: 1.5729429
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 35.00
Output dim: 2, lower bound: -1.5739630, upper bound: 1.5739617

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -8.9918203, -5.5807829, -8.9812546, -5.5830574, -3.0438175, 3.0322804
1: -6.5514622, -3.9748602, -6.5253983, -4.0035424, -2.4162283, 2.4179196
2: 8.3718681, 10.8819847, 8.3654623, 10.9080257, -2.4941034, 2.4767990
3: -6.0847559, -2.9110868, -6.0575352, -2.9428487, -3.1419072, 3.1464484
4: -11.8076391, -8.0095825, -11.8090429, -8.0168257, -3.3528161, 3.3629279
5: -13.6332169, -10.1961212, -13.6374359, -10.1905622, -2.9148684, 2.9133177
6: -15.6223717, -12.3353491, -15.6232204, -12.3340082, -2.7036881, 2.6837578
7: -5.5379381, -2.0711849, -5.5203929, -2.0657630, -3.4034843, 3.3884029
8: -1.9441729, 0.3750110, -1.9372301, 0.3551722, -2.2501101, 2.2511220
9: -7.2873898, -4.0344877, -7.2728357, -4.0552011, -2.9337077, 2.9305472

Time for backsubstitution: 14.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 6191
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 6111

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5666186, upper bound: 1.5532974
time: 5.31 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5694670, upper bound: 1.5533001
time: 11.29 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -8.9986115, -5.5798917, -9.0187511, -5.5666409, -3.0633793, 3.0774660
1: -6.5597706, -3.9739649, -6.5736513, -3.9632599, -2.4620147, 2.4582400
2: 8.3694992, 10.8838139, 8.3301048, 10.9183407, -2.5354958, 2.5078592
3: -6.0988588, -2.9099309, -6.1171198, -2.8889630, -3.2098958, 3.1776390
4: -11.8087730, -8.0043488, -11.8273125, -7.9849749, -3.3782234, 3.3890104
5: -13.6352243, -10.1946344, -13.6560535, -10.1831980, -2.9285965, 2.9380498
6: -15.6257639, -12.3342829, -15.6474380, -12.3202066, -2.7208943, 2.7113485
7: -5.5424218, -2.0679941, -5.5636358, -2.0532966, -3.4339027, 3.4298258
8: -1.9461651, 0.3789392, -1.9574494, 0.3830967, -2.2821369, 2.2804003
9: -7.2891226, -4.0242634, -7.3049965, -4.0083013, -2.9799356, 2.9886866

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6111

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5704821, upper bound: 1.5504548
time: 8.13 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5704861, upper bound: 1.5533000
time: 5.22 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -8.9886036, -5.5786982, -9.0192862, -5.5631227, -3.0931320, 3.0645952
1: -6.5282736, -3.9994304, -6.5681982, -3.9600048, -2.4365892, 2.4512584
2: 8.3598528, 10.9217129, 8.3268194, 10.9302473, -2.5309501, 2.5378094
3: -6.0636644, -2.9367185, -6.1091094, -2.8836012, -3.1800632, 3.1723909
4: -11.8151150, -8.0142670, -11.8322916, -7.9876604, -3.3880525, 3.3833504
5: -13.6450043, -10.1898537, -13.6616821, -10.1840324, -2.9435053, 2.9667649
6: -15.6315403, -12.3309956, -15.6523714, -12.3182526, -2.7316875, 2.7209878
7: -5.5253348, -2.0600867, -5.5641308, -2.0508013, -3.4217663, 3.4399920
8: -1.9408922, 0.3561544, -1.9591703, 0.3801689, -2.2764549, 2.2616911
9: -7.2787361, -4.0523582, -7.3092074, -4.0156713, -2.9493170, 2.9597774

Time for backsubstitution: 14.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 6191
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5533017, upper bound: 1.5694671
time: 7.76 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5533017, upper bound: 1.5729423
time: 6.78 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -9.0258732, -5.5622635, -9.0259018, -5.5622473, -3.1386814, 3.0830455
1: -6.5765219, -3.9591112, -6.5765362, -3.9590993, -2.4769516, 2.4968312
2: 8.3243294, 10.9320002, 8.3243122, 10.9320564, -2.5622711, 2.5808592
3: -6.1232567, -2.8826246, -6.1232810, -2.8826072, -3.2064366, 3.2406564
4: -11.8333616, -7.9824634, -11.8333826, -7.9824452, -3.4140463, 3.4087744
5: -13.6636276, -10.1825552, -13.6636505, -10.1825514, -2.9682226, 2.9804487
6: -15.6556339, -12.3172016, -15.6556597, -12.3171930, -2.7592654, 2.7380881
7: -5.5685940, -2.0476890, -5.5686159, -2.0476680, -3.4630704, 3.4702477
8: -1.9611812, 0.3840694, -1.9611945, 0.3840852, -2.3054781, 2.2933912
9: -7.3109002, -4.0054684, -7.3109226, -4.0054426, -3.0075564, 3.0059795

Time for backsubstitution: 14.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5533017, upper bound: 1.5704863
time: 7.72 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5533017, upper bound: 1.5739634
time: 5.36 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 28.03 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 28.03
Output dim: 2, lower bound: -1.5666186, upper bound: 1.5532974
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 28.03
Output dim: 2, lower bound: -1.5694670, upper bound: 1.5533001
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 28.03
Output dim: 2, lower bound: -1.5704821, upper bound: 1.5504548
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 28.03
Output dim: 2, lower bound: -1.5704861, upper bound: 1.5533000
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 28.03
Output dim: 2, lower bound: -1.5533017, upper bound: 1.5694671
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 28.03
Output dim: 2, lower bound: -1.5533017, upper bound: 1.5729423
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 28.03
Output dim: 2, lower bound: -1.5533017, upper bound: 1.5704863
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 28.03
Output dim: 2, lower bound: -1.5533017, upper bound: 1.5739634

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: -8.9906969, -5.5849228, -8.9621506, -5.5987740, -3.0267553, 3.0089760
1: -6.5492992, -3.9755118, -6.5110316, -4.0104713, -2.4062858, 2.4014533
2: 8.3726969, 10.8795414, 8.3798580, 10.8979607, -2.4796200, 2.4544671
3: -6.0822663, -2.9156833, -6.0331903, -2.9623270, -3.1199393, 3.1175070
4: -11.8051109, -8.0111561, -11.7990732, -8.0298414, -3.3312764, 3.3519635
5: -13.6314831, -10.1963902, -13.6282377, -10.1936474, -2.9094181, 2.9038243
6: -15.6218910, -12.3397655, -15.6097336, -12.3510780, -2.6853890, 2.6622605
7: -5.5276933, -2.0716789, -5.4786243, -2.0828767, -3.3760028, 3.3461323
8: -1.9433365, 0.3743749, -1.9298124, 0.3501687, -2.2401328, 2.2385645
9: -7.2858915, -4.0362134, -7.2651463, -4.0650730, -2.9218702, 2.9200029

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5494428, upper bound: 1.5532974
time: 5.00 seconds

## Relational analysis of IS_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5494428, upper bound: 1.5532972
time: 5.32 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: -8.9918175, -5.5807858, -8.9812489, -5.5830636, -3.0411711, 3.0322738
1: -6.5514593, -3.9748607, -6.5253925, -4.0035439, -2.4161768, 2.4152238
2: 8.3718691, 10.8819818, 8.3654652, 10.9080219, -2.4976673, 2.4668298
3: -6.0847535, -2.9110894, -6.0575280, -2.9428577, -3.1377897, 3.1464386
4: -11.8076391, -8.0095806, -11.8090382, -8.0168285, -3.3476276, 3.3626542
5: -13.6332140, -10.1961241, -13.6374321, -10.1905651, -2.9133759, 2.9133973
6: -15.6223717, -12.3353500, -15.6232204, -12.3340178, -2.6931314, 2.6809530
7: -5.5379324, -2.0711856, -5.5203700, -2.0657640, -3.4034739, 3.3794327
8: -1.9441733, 0.3750105, -1.9372287, 0.3551688, -2.2445140, 2.2543983
9: -7.2873883, -4.0344896, -7.2728291, -4.0552077, -2.9337001, 2.9312053

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5522911, upper bound: 1.5533001
time: 7.81 seconds

## Relational analysis of IS_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5522911, upper bound: 1.5533000
time: 5.03 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.9794273, -5.5956068, -9.0176210, -5.5707750, -3.0400066, 3.0603919
1: -6.5457754, -3.9809072, -6.5714760, -3.9639084, -2.4459367, 2.4482851
2: 8.3841810, 10.8737373, 8.3309422, 10.9158897, -2.5132537, 2.4934146
3: -6.0744781, -2.9294693, -6.1146431, -2.8935509, -3.1809273, 3.1557832
4: -11.7987986, -8.0174160, -11.8247814, -7.9865541, -3.3672190, 3.3675013
5: -13.6260395, -10.1977215, -13.6543293, -10.1834621, -2.9190788, 2.9326048
6: -15.6122198, -12.3513393, -15.6469517, -12.3246288, -2.6994953, 2.6930485
7: -5.5006495, -2.0851326, -5.5533834, -2.0537920, -3.3916349, 3.4023128
8: -1.9387493, 0.3739696, -1.9566221, 0.3824620, -2.2693071, 2.2703876
9: -7.2814326, -4.0340419, -7.3034964, -4.0100284, -2.9694142, 2.9769936

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 6191
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5533048, upper bound: 1.5504541
time: 6.99 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5533048, upper bound: 1.5504546
time: 5.36 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.9986086, -5.5799007, -9.0187492, -5.5666432, -3.0633717, 3.0748186
1: -6.5597639, -3.9739661, -6.5736489, -3.9632599, -2.4592938, 2.4582033
2: 8.3695011, 10.8838100, 8.3301077, 10.9183397, -2.5203524, 2.5113673
3: -6.0988512, -2.9099393, -6.1171184, -2.8889632, -3.2098880, 3.1623540
4: -11.8087702, -8.0043497, -11.8273106, -7.9849753, -3.3779287, 3.3838615
5: -13.6352186, -10.1946354, -13.6560535, -10.1831970, -2.9286804, 2.9365692
6: -15.6257601, -12.3342915, -15.6474380, -12.3202105, -2.7182226, 2.7007546
7: -5.5423994, -2.0679948, -5.5636301, -2.0532968, -3.4249320, 3.4298162
8: -1.9461632, 0.3789382, -1.9574485, 0.3830957, -2.2854228, 2.2747984
9: -7.2891178, -4.0242662, -7.3049960, -4.0083027, -2.9805965, 2.9886808

Time for backsubstitution: 14.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 6191
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5533073, upper bound: 1.5532997
time: 10.39 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5533073, upper bound: 1.5532998
time: 5.38 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -8.9886036, -5.5786982, -8.9918203, -5.5807829, -3.0297241, 3.0393834
1: -6.5282736, -3.9994304, -6.5514622, -3.9748602, -2.4161844, 2.4226823
2: 8.3598528, 10.9217129, 8.3718681, 10.8819847, -2.4802713, 2.5083528
3: -6.0636644, -2.9367185, -6.0847559, -2.9110868, -3.1525776, 3.1480374
4: -11.8151150, -8.0142670, -11.8076391, -8.0095825, -3.3696008, 3.3555002
5: -13.6450043, -10.1898537, -13.6332169, -10.1961212, -2.9262533, 2.9104776
6: -15.6315403, -12.3309956, -15.6223717, -12.3353491, -2.6911430, 2.6961036
7: -5.5253348, -2.0600867, -5.5379381, -2.0711849, -3.3931117, 3.4116282
8: -1.9408922, 0.3561544, -1.9441729, 0.3750110, -2.2517724, 2.2459269
9: -7.2787361, -4.0523582, -7.2873898, -4.0344877, -2.9372683, 2.9351392

Time for backsubstitution: 14.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6111

## Relational analysis of IS_A2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5532976, upper bound: 1.5666173
time: 18.57 seconds

## Relational analysis of IS_A2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5533001, upper bound: 1.5694657
time: 7.65 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -8.9886036, -5.5786982, -9.0192680, -5.5631351, -3.0930929, 3.1023884
1: -6.5282736, -3.9994304, -6.5681911, -3.9600179, -2.4432697, 2.4512336
2: 8.3598528, 10.9217129, 8.3268356, 10.9301968, -2.5219488, 2.5377998
3: -6.0636644, -2.9367185, -6.1090922, -2.8836174, -3.1800470, 3.1723738
4: -11.8151150, -8.0142670, -11.8322687, -7.9876652, -3.3880444, 3.3791227
5: -13.6450043, -10.1898537, -13.6616592, -10.1840324, -2.9580717, 2.9667363
6: -15.6315403, -12.3309956, -15.6523457, -12.3182640, -2.7316608, 2.7433066
7: -5.5253348, -2.0600867, -5.5641170, -2.0508201, -3.4192915, 3.4399672
8: -1.9408922, 0.3561544, -1.9591603, 0.3801665, -2.2764411, 2.2790437
9: -7.2787361, -4.0523582, -7.3091869, -4.0156798, -2.9493008, 2.9510455

Time for backsubstitution: 14.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 6191
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 6111

## Relational analysis of IS_A2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5532976, upper bound: 1.5700809
time: 9.35 seconds

## Relational analysis of IS_A2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5533001, upper bound: 1.5729422
time: 6.75 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -9.0258732, -5.5622635, -8.9986115, -5.5798917, -3.0754652, 3.0590158
1: -6.5765219, -3.9591112, -6.5597706, -3.9739649, -2.4565163, 2.4685340
2: 8.3243294, 10.9320002, 8.3694992, 10.8838139, -2.5116119, 2.5374088
3: -6.1232567, -2.8826246, -6.0988588, -2.9099309, -3.1829805, 3.2162342
4: -11.8333616, -7.9824634, -11.8087730, -8.0043488, -3.3956380, 3.3809171
5: -13.6636276, -10.1825552, -13.6352243, -10.1946344, -2.9509745, 2.9241805
6: -15.6556339, -12.3172016, -15.6257639, -12.3342829, -2.7188554, 2.7132826
7: -5.5685940, -2.0476890, -5.5424218, -2.0679941, -3.4344988, 3.4420810
8: -1.9611812, 0.3840694, -1.9461651, 0.3789392, -2.2807093, 2.2778616
9: -7.3109002, -4.0054684, -7.2891226, -4.0242634, -2.9954815, 2.9813347

Time for backsubstitution: 14.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 93

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 6111

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5504546, upper bound: 1.5704809
time: 7.48 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5533000, upper bound: 1.5704850
time: 14.40 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -9.0258732, -5.5622635, -9.0258808, -5.5622616, -3.1386433, 3.1208410
1: -6.5765219, -3.9591112, -6.5765285, -3.9591107, -2.4834309, 2.4968066
2: 8.3243294, 10.9320002, 8.3243237, 10.9320040, -2.5532703, 2.5750408
3: -6.1232567, -2.8826246, -6.1232624, -2.8826249, -3.2286701, 3.2406378
4: -11.8333616, -7.9824634, -11.8333626, -7.9824505, -3.4140406, 3.4045458
5: -13.6636276, -10.1825552, -13.6636324, -10.1825542, -2.9827871, 2.9804201
6: -15.6556339, -12.3172016, -15.6556358, -12.3172035, -2.7592363, 2.7605238
7: -5.5685940, -2.0476890, -5.5686007, -2.0476868, -3.4608960, 3.4702229
8: -1.9611812, 0.3840694, -1.9611821, 0.3840814, -2.3054633, 2.3107438
9: -7.3109002, -4.0054684, -7.3109016, -4.0054512, -3.0075412, 2.9973125

Time for backsubstitution: 14.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 6111

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5504546, upper bound: 1.5739572
time: 7.50 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5533000, upper bound: 1.5739618
time: 6.42 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 29.14 seconds
IS_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 29.14
Output dim: 2, lower bound: -1.5494428, upper bound: 1.5532974
IS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 29.14
Output dim: 2, lower bound: -1.5494428, upper bound: 1.5532972
IS_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 29.14
Output dim: 2, lower bound: -1.5522911, upper bound: 1.5533001
IS_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 29.14
Output dim: 2, lower bound: -1.5522911, upper bound: 1.5533000
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.14
Output dim: 2, lower bound: -1.5533048, upper bound: 1.5504541
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.14
Output dim: 2, lower bound: -1.5533048, upper bound: 1.5504546
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.14
Output dim: 2, lower bound: -1.5533073, upper bound: 1.5532997
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.14
Output dim: 2, lower bound: -1.5533073, upper bound: 1.5532998
IS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 29.14
Output dim: 2, lower bound: -1.5532976, upper bound: 1.5666173
IS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 29.14
Output dim: 2, lower bound: -1.5533001, upper bound: 1.5694657
IS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 29.14
Output dim: 2, lower bound: -1.5532976, upper bound: 1.5700809
IS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 29.14
Output dim: 2, lower bound: -1.5533001, upper bound: 1.5729422
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 29.14
Output dim: 2, lower bound: -1.5504546, upper bound: 1.5704809
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 29.14
Output dim: 2, lower bound: -1.5533000, upper bound: 1.5704850
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 29.14
Output dim: 2, lower bound: -1.5504546, upper bound: 1.5739572
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 29.14
Output dim: 2, lower bound: -1.5533000, upper bound: 1.5739618

## BFS IS instance: IS_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -8.9906969, -5.5849228, -8.9424400, -5.6120148, -2.9997931, 2.9866481
1: -6.5492992, -3.9755118, -6.4981365, -4.0209436, -2.3928328, 2.3791170
2: 8.3726969, 10.8795414, 8.4179306, 10.8633242, -2.4431019, 2.4206583
3: -6.0822663, -2.9156833, -6.0149341, -2.9832096, -3.0990567, 3.0992508
4: -11.8051109, -8.0111561, -11.7803307, -8.0491629, -3.3113403, 3.3307176
5: -13.6314831, -10.1963902, -13.6074848, -10.2052059, -2.8922529, 2.8797121
6: -15.6218910, -12.3397655, -15.5885983, -12.3651686, -2.6598597, 2.6458344
7: -5.5276933, -2.0716789, -5.4573860, -2.0976663, -3.3558269, 3.3203111
8: -1.9433365, 0.3743749, -1.9193792, 0.3459344, -2.2295837, 2.2251725
9: -7.2858915, -4.0362134, -7.2492390, -4.0804982, -2.9029589, 2.9022002

Time for backsubstitution: 14.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5494360, upper bound: 1.5494369
time: 12.19 seconds

## Relational analysis of IS_A1_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5494360, upper bound: 1.5532909
time: 5.39 seconds

## BFS IS instance: IS_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -8.9906969, -5.5849228, -8.9694948, -5.5944099, -3.0223198, 3.0065017
1: -6.5492992, -3.9755118, -6.5139165, -4.0063586, -2.4127398, 2.3997006
2: 8.3726969, 10.8795414, 8.3742714, 10.9116440, -2.4938712, 2.4579227
3: -6.0822663, -2.9156833, -6.0393186, -2.9561229, -3.1261435, 3.1236353
4: -11.8051109, -8.0111561, -11.8051472, -8.0272884, -3.3339643, 3.3586383
5: -13.6314831, -10.1963902, -13.6358128, -10.1929359, -2.9050264, 2.9167538
6: -15.6218910, -12.3397655, -15.6180477, -12.3480673, -2.6778030, 2.6697211
7: -5.5276933, -2.0716789, -5.4835715, -2.0771976, -3.3841562, 3.3508410
8: -1.9433365, 0.3743749, -1.9334793, 0.3511448, -2.2359452, 2.2393208
9: -7.2858915, -4.0362134, -7.2710423, -4.0622311, -2.9232965, 2.9267225

Time for backsubstitution: 14.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5494378, upper bound: 1.5494367
time: 5.76 seconds

## Relational analysis of IS_A1_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5494360, upper bound: 1.5532907
time: 5.27 seconds

## BFS IS instance: IS_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -8.9918175, -5.5807858, -8.9608021, -5.5963144, -3.0142269, 3.0091057
1: -6.5514593, -3.9748607, -6.5115428, -4.0140209, -2.4027405, 2.3915937
2: 8.3718691, 10.8819818, 8.4046631, 10.8733797, -2.4611487, 2.4320166
3: -6.0847535, -2.9110894, -6.0392895, -2.9645543, -3.1160631, 3.1282001
4: -11.8076391, -8.0095806, -11.7903070, -8.0363884, -3.3274412, 3.3414173
5: -13.6332140, -10.1961241, -13.6165075, -10.2021227, -2.8962097, 2.8890910
6: -15.6223717, -12.3353500, -15.6012774, -12.3481092, -2.6675930, 2.6636019
7: -5.5379324, -2.0711856, -5.4991269, -2.0805755, -3.3832655, 3.3535943
8: -1.9441733, 0.3750105, -1.9262028, 0.3509288, -2.2339611, 2.2401667
9: -7.2873883, -4.0344896, -7.2569127, -4.0712085, -2.9143038, 2.9133968

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5522844, upper bound: 1.5494375
time: 5.80 seconds

## Relational analysis of IS_A1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5522844, upper bound: 1.5532935
time: 11.49 seconds

## BFS IS instance: IS_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -8.9918175, -5.5807858, -8.9885998, -5.5787039, -3.0367365, 3.0297184
1: -6.5514593, -3.9748607, -6.5282674, -3.9994316, -2.4226313, 2.4134881
2: 8.3718691, 10.8819818, 8.3598566, 10.9217072, -2.5060697, 2.4703183
3: -6.0847535, -2.9110894, -6.0636568, -2.9367275, -3.1330681, 3.1525674
4: -11.8076391, -8.0095806, -11.8151102, -8.0142717, -3.3503156, 3.3693266
5: -13.6332140, -10.1961241, -13.6449966, -10.1898537, -2.9089851, 2.9263339
6: -15.6223717, -12.3353500, -15.6315413, -12.3310051, -2.6855478, 2.6883426
7: -5.5379324, -2.0711856, -5.5253115, -2.0600872, -3.4116187, 3.3841400
8: -1.9441733, 0.3750105, -1.9408894, 0.3561511, -2.2403336, 2.2550483
9: -7.2873883, -4.0344896, -7.2787313, -4.0523629, -2.9351306, 2.9379253

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5522844, upper bound: 1.5494374
time: 5.20 seconds

## Relational analysis of IS_A1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5522844, upper bound: 1.5532935
time: 5.09 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.9794273, -5.5956068, -8.9974842, -5.5840344, -3.0131502, 3.0359821
1: -6.5457754, -3.9809072, -6.5575905, -3.9746170, -2.4320326, 2.4247732
2: 8.3841810, 10.8737373, 8.3703251, 10.8813629, -2.4789782, 2.4580572
3: -6.0744781, -2.9294693, -6.0963612, -2.9145288, -3.1599493, 3.1323571
4: -11.7987986, -8.0174160, -11.8062439, -8.0059319, -3.3472996, 3.3462853
5: -13.6260395, -10.1977215, -13.6334867, -10.1949005, -2.9020042, 2.9084444
6: -15.6122198, -12.3513393, -15.6252804, -12.3387032, -2.6739736, 2.6757903
7: -5.5006495, -2.0851326, -5.5321717, -2.0684929, -3.3712583, 3.3763604
8: -1.9387493, 0.3739696, -1.9453287, 0.3782921, -2.2588162, 2.2558122
9: -7.2814326, -4.0340419, -7.2876205, -4.0260086, -2.9500732, 2.9591541

Time for backsubstitution: 14.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 6191
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5532981, upper bound: 1.5465968
time: 6.81 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5532981, upper bound: 1.5504482
time: 6.86 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.9794273, -5.5956068, -9.0247459, -5.5663924, -3.0356474, 3.0583963
1: -6.5457754, -3.9809072, -6.5743494, -3.9597611, -2.4524565, 2.4465568
2: 8.3841810, 10.8737373, 8.3251724, 10.9295502, -2.5151672, 2.4971635
3: -6.0744781, -2.9294693, -6.1207848, -2.8872156, -3.1872625, 3.1611290
4: -11.7987986, -8.0174160, -11.8308334, -7.9840407, -3.3699074, 3.3741279
5: -13.6260395, -10.1977215, -13.6619091, -10.1828232, -2.9146624, 2.9455280
6: -15.6122198, -12.3513393, -15.6551437, -12.3216209, -2.6918826, 2.7005563
7: -5.5006495, -2.0851326, -5.5583458, -2.0481849, -3.3998175, 3.4069757
8: -1.9387493, 0.3739696, -1.9603539, 0.3834362, -2.2650313, 2.2707005
9: -7.2814326, -4.0340419, -7.3093967, -4.0071955, -2.9708190, 2.9837899

Time for backsubstitution: 14.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 6191
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5532981, upper bound: 1.5465966
time: 5.22 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5532981, upper bound: 1.5504480
time: 6.91 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.9986086, -5.5799007, -8.9986029, -5.5798965, -3.0365076, 3.0503988
1: -6.5597639, -3.9739661, -6.5597630, -3.9739671, -2.4453993, 2.4347010
2: 8.3695011, 10.8838100, 8.3695030, 10.8838100, -2.4914536, 2.4760695
3: -6.0988512, -2.9099393, -6.0988498, -2.9099348, -3.1889164, 3.1389427
4: -11.8087702, -8.0043497, -11.8087730, -8.0043573, -3.3580117, 3.3626456
5: -13.6352186, -10.1946354, -13.6352186, -10.1946373, -2.9116020, 2.9124336
6: -15.6257601, -12.3342915, -15.6257591, -12.3342857, -2.6926975, 2.6835341
7: -5.5423994, -2.0679948, -5.5424118, -2.0679972, -3.4045515, 3.4038534
8: -1.9461632, 0.3789382, -1.9461622, 0.3789277, -2.2749333, 2.2602167
9: -7.2891178, -4.0242662, -7.2891188, -4.0242805, -2.9612808, 2.9708433

Time for backsubstitution: 14.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 6191
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5533006, upper bound: 1.5494371
time: 22.59 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5533006, upper bound: 1.5532935
time: 8.87 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.9986086, -5.5799007, -9.0258722, -5.5622644, -3.0590076, 3.0728188
1: -6.5597639, -3.9739661, -6.5765200, -3.9591107, -2.4658136, 2.4564791
2: 8.3695011, 10.8838100, 8.3243303, 10.9319992, -2.5222650, 2.5089748
3: -6.0988512, -2.9099393, -6.1232548, -2.8826261, -3.2131262, 3.1676970
4: -11.8087702, -8.0043497, -11.8333616, -7.9824638, -3.3806210, 3.3890581
5: -13.6352186, -10.1946354, -13.6636276, -10.1825562, -2.9242620, 2.9494777
6: -15.6257601, -12.3342915, -15.6556320, -12.3172045, -2.7093458, 2.7082582
7: -5.5423994, -2.0679948, -5.5685902, -2.0476892, -3.4331102, 3.4344888
8: -1.9461632, 0.3789382, -1.9611802, 0.3840690, -2.2811441, 2.2751083
9: -7.2891178, -4.0242662, -7.3108988, -4.0054712, -2.9819946, 2.9954767

Time for backsubstitution: 14.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 6191
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5533006, upper bound: 1.5494374
time: 5.05 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5533006, upper bound: 1.5532934
time: 4.77 seconds

## BFS IS instance: IS_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.9694948, -5.5944099, -8.9906969, -5.5849228, -3.0065022, 3.0223193
1: -6.5139165, -4.0063586, -6.5492992, -3.9755118, -2.3997002, 2.4127393
2: 8.3742714, 10.9116440, 8.3726969, 10.8795414, -2.4579225, 2.4938712
3: -6.0393186, -2.9561229, -6.0822663, -2.9156833, -3.1236353, 3.1261435
4: -11.8051472, -8.0272884, -11.8051109, -8.0111561, -3.3586383, 3.3339653
5: -13.6358128, -10.1929359, -13.6314831, -10.1963902, -2.9167538, 2.9050264
6: -15.6180477, -12.3480673, -15.6218910, -12.3397655, -2.6697216, 2.6778030
7: -5.4835715, -2.0771976, -5.5276933, -2.0716789, -3.3508410, 3.3841562
8: -1.9334793, 0.3511448, -1.9433365, 0.3743749, -2.2393208, 2.2359452
9: -7.2710423, -4.0622311, -7.2858915, -4.0362134, -2.9267225, 2.9232965

Time for backsubstitution: 14.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_A2_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5494372, upper bound: 1.5666105
time: 9.84 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5532911, upper bound: 1.5666103
time: 8.87 seconds

## BFS IS instance: IS_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.9885998, -5.5787039, -8.9918175, -5.5807858, -3.0297194, 3.0367365
1: -6.5282674, -3.9994316, -6.5514593, -3.9748607, -2.4134884, 2.4226315
2: 8.3598566, 10.9217072, 8.3718691, 10.8819818, -2.4703183, 2.5060694
3: -6.0636568, -2.9367275, -6.0847535, -2.9110894, -3.1525674, 3.1330686
4: -11.8151102, -8.0142717, -11.8076391, -8.0095806, -3.3693266, 3.3503160
5: -13.6449966, -10.1898537, -13.6332140, -10.1961241, -2.9263334, 2.9089847
6: -15.6315413, -12.3310051, -15.6223717, -12.3353500, -2.6883421, 2.6855474
7: -5.5253115, -2.0600872, -5.5379324, -2.0711856, -3.3841400, 3.4116192
8: -1.9408894, 0.3561511, -1.9441733, 0.3750105, -2.2550488, 2.2403336
9: -7.2787313, -4.0523629, -7.2873883, -4.0344896, -2.9379258, 2.9351311

Time for backsubstitution: 14.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 93

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_A2_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5494376, upper bound: 1.5694594
time: 10.43 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5532954, upper bound: 1.5694589
time: 8.41 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 34.00 seconds
IS_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 34.00
Output dim: 2, lower bound: -1.5494360, upper bound: 1.5494369
IS_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 34.00
Output dim: 2, lower bound: -1.5494360, upper bound: 1.5532909
IS_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 34.00
Output dim: 2, lower bound: -1.5494378, upper bound: 1.5494367
IS_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 34.00
Output dim: 2, lower bound: -1.5494360, upper bound: 1.5532907
IS_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 34.00
Output dim: 2, lower bound: -1.5522844, upper bound: 1.5494375
IS_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 34.00
Output dim: 2, lower bound: -1.5522844, upper bound: 1.5532935
IS_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 34.00
Output dim: 2, lower bound: -1.5522844, upper bound: 1.5494374
IS_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 34.00
Output dim: 2, lower bound: -1.5522844, upper bound: 1.5532935
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 34.00
Output dim: 2, lower bound: -1.5532981, upper bound: 1.5465968
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 34.00
Output dim: 2, lower bound: -1.5532981, upper bound: 1.5504482
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 34.00
Output dim: 2, lower bound: -1.5532981, upper bound: 1.5465966
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 34.00
Output dim: 2, lower bound: -1.5532981, upper bound: 1.5504480
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 34.00
Output dim: 2, lower bound: -1.5533006, upper bound: 1.5494371
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 34.00
Output dim: 2, lower bound: -1.5533006, upper bound: 1.5532935
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 34.00
Output dim: 2, lower bound: -1.5533006, upper bound: 1.5494374
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 34.00
Output dim: 2, lower bound: -1.5533006, upper bound: 1.5532934
IS_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 34.00
Output dim: 2, lower bound: -1.5494372, upper bound: 1.5666105
IS_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 34.00
Output dim: 2, lower bound: -1.5532911, upper bound: 1.5666103
IS_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 34.00
Output dim: 2, lower bound: -1.5494376, upper bound: 1.5694594
IS_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 34.00
Output dim: 2, lower bound: -1.5532954, upper bound: 1.5694589
IS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 34.00
Output dim: 2, lower bound: -1.5532976, upper bound: 1.5700809
IS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 34.00
Output dim: 2, lower bound: -1.5533001, upper bound: 1.5729422
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 34.00
Output dim: 2, lower bound: -1.5504546, upper bound: 1.5704809
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 34.00
Output dim: 2, lower bound: -1.5533000, upper bound: 1.5704850
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 34.00
Output dim: 2, lower bound: -1.5504546, upper bound: 1.5739572
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 34.00
Output dim: 2, lower bound: -1.5533000, upper bound: 1.5739618
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.579909086227417
rel_dist={2: [-1.574020518019939, 1.5740203796460204]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 6191
type: B, layer: 1, pos: 6191
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714664, upper bound: 1.2809088
time: 45.83 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2848991, upper bound: 1.2849000
time: 20.15 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 66.24 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 66.24
Output dim: 2, lower bound: -1.2714664, upper bound: 1.2809088
IS_B2, status: Status.UNKNOWN, split count: 1, time: 66.24
Output dim: 2, lower bound: -1.2848991, upper bound: 1.2849000

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -9.0141144, -5.5696034, -8.9986153, -5.5798907, -2.7948151, 2.7963891
1: -6.5716915, -3.9660313, -6.5597734, -3.9739642, -2.2566381, 2.2454674
2: 8.3341942, 10.9090824, 8.3694963, 10.8838148, -2.2863641, 2.2806928
3: -6.1129909, -2.8935089, -6.0988617, -2.9099305, -2.9742832, 2.9739122
4: -11.8232069, -7.9867201, -11.8087769, -8.0043459, -3.0597548, 3.0611587
5: -13.6509228, -10.1836557, -13.6352262, -10.1946335, -2.5796614, 2.5792108
6: -15.6420031, -12.3222532, -15.6257658, -12.3342810, -2.3928723, 2.3998499
7: -5.5602894, -2.0570686, -5.5424242, -2.0679922, -3.2681723, 3.2616634
8: -1.9549789, 0.3824530, -1.9461651, 0.3789444, -2.1109905, 2.1077843
9: -7.3010244, -4.0102215, -7.2891226, -4.0242567, -2.7538757, 2.7575927

Time for backsubstitution: 14.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6191
type: B, layer: 1, pos: 6191
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 6191

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714662, upper bound: 1.2805264
time: 9.17 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714663, upper bound: 1.2809084
time: 8.41 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -9.0258961, -5.5622540, -9.0258865, -5.5622592, -2.8170185, 2.8530989
1: -6.5765371, -3.9591014, -6.5765324, -3.9591095, -2.2823482, 2.2782660
2: 8.3243179, 10.9320374, 8.3243237, 10.9320021, -2.3325753, 2.3431334
3: -6.1232762, -2.8826139, -6.1232662, -2.8826249, -3.0259595, 3.0041327
4: -11.8333778, -7.9824438, -11.8333654, -7.9824467, -3.0935006, 3.0885391
5: -13.6636486, -10.1825542, -13.6636353, -10.1825542, -2.6297121, 2.6154127
6: -15.6556530, -12.3171959, -15.6556358, -12.3172054, -2.4192305, 2.4359670
7: -5.5686121, -2.0476735, -5.5686011, -2.0476840, -3.3023171, 3.3057709
8: -1.9611921, 0.3840876, -1.9611835, 0.3840857, -2.1189008, 2.1352043
9: -7.3109159, -4.0054383, -7.3109016, -4.0054460, -2.7860365, 2.7748094

Time for backsubstitution: 14.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6191
type: A, layer: 1, pos: 6191
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 4639
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6191

## Relational analysis of IS_B2_B1

### Relational analysis result of IS_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2843670, upper bound: 1.2848998
time: 9.36 seconds

## Relational analysis of IS_B2_B2

### Relational analysis result of IS_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2848989, upper bound: 1.2848995
time: 8.50 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 32.99 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 32.99
Output dim: 2, lower bound: -1.2714662, upper bound: 1.2805264
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 32.99
Output dim: 2, lower bound: -1.2714663, upper bound: 1.2809084
IS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 32.99
Output dim: 2, lower bound: -1.2843670, upper bound: 1.2848998
IS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 32.99
Output dim: 2, lower bound: -1.2848989, upper bound: 1.2848995

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -8.9764557, -5.5860114, -8.9866161, -5.5815153, -2.7549400, 2.7630215
1: -6.5234237, -4.0062799, -6.5447364, -3.9755635, -2.2019720, 2.1923892
2: 8.3694496, 10.8987541, 8.3738289, 10.8805056, -2.2357292, 2.2449939
3: -6.0534167, -2.9472728, -6.0733662, -2.9120522, -2.9110479, 2.8918281
4: -11.8049183, -8.0186052, -11.8066998, -8.0137587, -3.0293336, 3.0251746
5: -13.6323032, -10.1910591, -13.6315594, -10.1972256, -2.5560074, 2.5625839
6: -15.6176958, -12.3360510, -15.6198521, -12.3361921, -2.3631477, 2.3801951
7: -5.5170918, -2.0695891, -5.5343361, -2.0729911, -3.2162476, 3.2275167
8: -1.9348083, 0.3545079, -1.9425387, 0.3718724, -2.0744252, 2.0734825
9: -7.2688732, -4.0571499, -7.2860117, -4.0427942, -2.6954184, 2.7078018

Time for backsubstitution: 14.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 6111

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714653, upper bound: 1.2788191
time: 12.58 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714653, upper bound: 1.2805255
time: 7.31 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -9.0141048, -5.5696077, -8.9986076, -5.5798912, -2.7942200, 2.7846055
1: -6.5716820, -3.9660320, -6.5597677, -3.9739640, -2.2408862, 2.2454538
2: 8.3341999, 10.9090776, 8.3695002, 10.8838139, -2.2645092, 2.2813747
3: -6.1129808, -2.8935108, -6.0988569, -2.9099307, -2.9175472, 2.9720502
4: -11.8232031, -7.9867334, -11.8087730, -8.0043507, -3.0597458, 3.0499835
5: -13.6509132, -10.1836557, -13.6352234, -10.1946354, -2.5814247, 2.5786266
6: -15.6419983, -12.3222542, -15.6257620, -12.3342838, -2.3913574, 2.3998456
7: -5.5602818, -2.0570741, -5.5424213, -2.0679944, -3.2543974, 3.2571278
8: -1.9549751, 0.3824377, -1.9461637, 0.3789368, -2.1050820, 2.1046286
9: -7.3010216, -4.0102444, -7.2891221, -4.0242658, -2.7614970, 2.7550941

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 6191
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6111

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2697925, upper bound: 1.2809073
time: 8.02 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714653, upper bound: 1.2809076
time: 27.72 seconds

## BFS IS instance: IS_B2_B1

### Backsubstitution after applying IS history:
0: -9.0142241, -5.5638433, -8.9886036, -5.5786982, -2.7841673, 2.8124185
1: -6.5614448, -3.9607232, -6.5282736, -3.9994304, -2.2294817, 2.2235899
2: 8.3288994, 10.9287720, 8.3598528, 10.9217129, -2.2966361, 2.2923207
3: -6.0976601, -2.8844519, -6.0636644, -2.9367185, -2.9435277, 2.9407187
4: -11.8313808, -7.9918265, -11.8151150, -8.0142670, -3.0574956, 3.0582705
5: -13.6600494, -10.1851320, -13.6450043, -10.1898537, -2.6131678, 2.5918117
6: -15.6499233, -12.3190918, -15.6315403, -12.3309956, -2.3997464, 2.4062657
7: -5.5605273, -2.0525689, -5.5253348, -2.0600867, -3.2681103, 3.2540412
8: -1.9575086, 0.3770390, -1.9408922, 0.3561544, -2.0847297, 2.0989203
9: -7.3078370, -4.0239811, -7.2787361, -4.0523582, -2.7362885, 2.7162018

Time for backsubstitution: 14.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4639
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 6191
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 6111

## Relational analysis of IS_B2_B1_B1

### Relational analysis result of IS_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2826483, upper bound: 1.2848987
time: 5.83 seconds

## Relational analysis of IS_B2_B1_B2

### Relational analysis result of IS_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2843660, upper bound: 1.2849012
time: 9.36 seconds

## BFS IS instance: IS_B2_B2

### Backsubstitution after applying IS history:
0: -9.0258923, -5.5622559, -9.0258732, -5.5622635, -2.8044729, 2.8524008
1: -6.5765319, -3.9591036, -6.5765219, -3.9591112, -2.2823343, 2.2625148
2: 8.3243179, 10.9320354, 8.3243294, 10.9320002, -2.3352900, 2.3192234
3: -6.1232724, -2.8826127, -6.1232567, -2.8826246, -3.0163488, 2.9467149
4: -11.8333750, -7.9824500, -11.8333616, -7.9824634, -3.0823240, 3.0885301
5: -13.6636419, -10.1825523, -13.6636276, -10.1825552, -2.6291313, 2.6171775
6: -15.6556520, -12.3171959, -15.6556339, -12.3172016, -2.4192271, 2.4344501
7: -5.5686083, -2.0476751, -5.5685940, -2.0476890, -3.2976055, 3.2920918
8: -1.9611893, 0.3840795, -1.9611812, 0.3840694, -2.1155052, 2.1293941
9: -7.3109140, -4.0054507, -7.3109002, -4.0054684, -2.7835369, 2.7824316

Time for backsubstitution: 14.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4639
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 6191
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 6111

## Relational analysis of IS_B2_B2_A1

### Relational analysis result of IS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2848979, upper bound: 1.2831871
time: 7.96 seconds

## Relational analysis of IS_B2_B2_A2

### Relational analysis result of IS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2848979, upper bound: 1.2849011
time: 9.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 32.32 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 32.32
Output dim: 2, lower bound: -1.2714653, upper bound: 1.2788191
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 32.32
Output dim: 2, lower bound: -1.2714653, upper bound: 1.2805255
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 32.32
Output dim: 2, lower bound: -1.2697925, upper bound: 1.2809073
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 32.32
Output dim: 2, lower bound: -1.2714653, upper bound: 1.2809076
IS_B2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 32.32
Output dim: 2, lower bound: -1.2826483, upper bound: 1.2848987
IS_B2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 32.32
Output dim: 2, lower bound: -1.2843660, upper bound: 1.2849012
IS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 32.32
Output dim: 2, lower bound: -1.2848979, upper bound: 1.2831871
IS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 32.32
Output dim: 2, lower bound: -1.2848979, upper bound: 1.2849011

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -8.9573584, -5.6017294, -8.9845791, -5.5890093, -2.7282372, 2.7450776
1: -6.5090499, -4.0132074, -6.5408258, -3.9767387, -2.1847796, 2.1805575
2: 8.3838062, 10.8886871, 8.3753386, 10.8760891, -2.2127223, 2.2296889
3: -6.0290747, -2.9667430, -6.0688381, -2.9203925, -2.8789868, 2.8679190
4: -11.7949457, -8.0316143, -11.8021212, -8.0166149, -3.0169067, 3.0023937
5: -13.6230974, -10.1941414, -13.6284189, -10.1977081, -2.5461426, 2.5563784
6: -15.6042061, -12.3531237, -15.6189842, -12.3441916, -2.3379450, 2.3610578
7: -5.4753194, -2.0867023, -5.5157828, -2.0738883, -3.1736250, 3.1915898
8: -1.9273872, 0.3495083, -1.9410219, 0.3706784, -2.0610356, 2.0630703
9: -7.2611876, -4.0670204, -7.2833052, -4.0459261, -2.6838245, 2.6944695

Time for backsubstitution: 14.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 93

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_B1_A1_A1_A1

### Relational analysis result of IS_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714617, upper bound: 1.2767624
time: 9.79 seconds

## Relational analysis of IS_B1_A1_A1_A2

### Relational analysis result of IS_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714617, upper bound: 1.2788172
time: 13.60 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -8.9764490, -5.5860186, -8.9866142, -5.5815206, -2.7549305, 2.7599068
1: -6.5234175, -4.0062819, -6.5447326, -3.9755645, -2.1988049, 2.1923366
2: 8.3694506, 10.8987484, 8.3738308, 10.8805017, -2.2242894, 2.2452977
3: -6.0534096, -2.9472823, -6.0733624, -2.9120555, -2.9110327, 2.8738408
4: -11.8049145, -8.0186090, -11.8066969, -8.0137615, -3.0269527, 3.0199804
5: -13.6323004, -10.1910563, -13.6315575, -10.1972275, -2.5553608, 2.5610580
6: -15.6176910, -12.3360615, -15.6198521, -12.3361959, -2.3603396, 2.3672285
7: -5.5170698, -2.0695889, -5.5343256, -2.0729918, -3.2056961, 3.2275000
8: -1.9348068, 0.3545051, -1.9425373, 0.3718696, -2.0763545, 2.0666547
9: -7.2688689, -4.0571542, -7.2860103, -4.0427985, -2.6957922, 2.7077932

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 6191
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2693883, upper bound: 1.2805221
time: 10.27 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714616, upper bound: 1.2805216
time: 8.85 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.0120564, -5.5770893, -8.9794235, -5.5956059, -2.7762747, 2.7578392
1: -6.5677328, -3.9672124, -6.5457721, -3.9809070, -2.2290316, 2.2286808
2: 8.3357143, 10.9046412, 8.3841839, 10.8737373, -2.2492700, 2.2584696
3: -6.1084728, -2.9018395, -6.0744758, -2.9294696, -2.8940172, 2.9372563
4: -11.8186235, -7.9896059, -11.7987976, -8.0174179, -3.0369883, 3.0375175
5: -13.6477861, -10.1841373, -13.6260395, -10.1977196, -2.5752287, 2.5687385
6: -15.6411171, -12.3302555, -15.6122169, -12.3513412, -2.3722124, 2.3747349
7: -5.5417204, -2.0579729, -5.5006485, -2.0851355, -3.2184348, 3.2145143
8: -1.9534755, 0.3812504, -1.9387484, 0.3739662, -2.0946383, 2.0910821
9: -7.2983084, -4.0133843, -7.2814322, -4.0340481, -2.7483087, 2.7435064

Time for backsubstitution: 14.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 6191
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_B1_A2_B1_B1

### Relational analysis result of IS_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2677201, upper bound: 1.2809039
time: 11.25 seconds

## Relational analysis of IS_B1_A2_B1_B2

### Relational analysis result of IS_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2697889, upper bound: 1.2809036
time: 9.29 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.0141010, -5.5696106, -8.9986038, -5.5799017, -2.7911062, 2.7845955
1: -6.5716782, -3.9660335, -6.5597615, -3.9739671, -2.2408481, 2.2422493
2: 8.3342009, 10.9090767, 8.3695021, 10.8838081, -2.2647805, 2.2650249
3: -6.1129775, -2.8935146, -6.0988488, -2.9099398, -2.8995609, 2.9615078
4: -11.8232012, -7.9867334, -11.8087711, -8.0043545, -3.0523028, 3.0475783
5: -13.6509142, -10.1836548, -13.6352205, -10.1946363, -2.5799532, 2.5779810
6: -15.6419973, -12.3222618, -15.6257591, -12.3342915, -2.3783631, 2.3971720
7: -5.5602694, -2.0570745, -5.5423985, -2.0679963, -3.2543783, 3.2465754
8: -1.9549742, 0.3824372, -1.9461617, 0.3789349, -2.0982494, 2.1065774
9: -7.3010187, -4.0102472, -7.2891169, -4.0242739, -2.7614889, 2.7554655

Time for backsubstitution: 14.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_B1_A2_B2_B1

### Relational analysis result of IS_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2693883, upper bound: 1.2809039
time: 8.11 seconds

## Relational analysis of IS_B1_A2_B2_B2

### Relational analysis result of IS_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714617, upper bound: 1.2809058
time: 7.20 seconds

## BFS IS instance: IS_B2_B1_B1

### Backsubstitution after applying IS history:
0: -9.0121803, -5.5713129, -8.9694948, -5.5944099, -2.7662067, 2.7856841
1: -6.5575361, -3.9618886, -6.5139165, -4.0063586, -2.2176714, 2.2063961
2: 8.3304424, 10.9243546, 8.3742714, 10.9116440, -2.2813144, 2.2692940
3: -6.0931625, -2.8927822, -6.0393186, -2.9561229, -2.9196424, 2.9086928
4: -11.8268003, -7.9947004, -11.8051472, -8.0272884, -3.0347147, 3.0458317
5: -13.6569309, -10.1856136, -13.6358128, -10.1929359, -2.6069574, 2.5819402
6: -15.6490402, -12.3270988, -15.6180477, -12.3480673, -2.3805943, 2.3810539
7: -5.5419655, -2.0534666, -5.4835715, -2.0771976, -3.2321830, 3.2114258
8: -1.9560051, 0.3758516, -1.9334793, 0.3511448, -2.0743051, 2.0856457
9: -7.3051238, -4.0271053, -7.2710423, -4.0622311, -2.7229662, 2.7046442

Time for backsubstitution: 14.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4639
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_B2_B1_B1_B1

### Relational analysis result of IS_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2806092, upper bound: 1.2848952
time: 5.87 seconds

## Relational analysis of IS_B2_B1_B1_B2

### Relational analysis result of IS_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2826447, upper bound: 1.2848951
time: 5.85 seconds

## BFS IS instance: IS_B2_B1_B2

### Backsubstitution after applying IS history:
0: -9.0142231, -5.5638471, -8.9885998, -5.5787039, -2.7810526, 2.8124075
1: -6.5614429, -3.9607239, -6.5282674, -3.9994316, -2.2294288, 2.2204173
2: 8.3289013, 10.9287720, 8.3598566, 10.9217072, -2.2968559, 2.2786109
3: -6.0976572, -2.8844562, -6.0636568, -2.9367275, -2.9255409, 2.9407043
4: -11.8313789, -7.9918280, -11.8151102, -8.0142717, -3.0523086, 3.0558786
5: -13.6600494, -10.1851301, -13.6449966, -10.1898537, -2.6116009, 2.5911674
6: -15.6499224, -12.3190975, -15.6315413, -12.3310051, -2.3867311, 2.4012733
7: -5.5605159, -2.0525708, -5.5253115, -2.0600872, -3.2680912, 3.2434874
8: -1.9575071, 0.3770380, -1.9408894, 0.3561511, -2.0779061, 2.1008449
9: -7.3078356, -4.0239859, -7.2787313, -4.0523629, -2.7362814, 2.7165747

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 4639
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 6191
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_B2_B1_B2_A1

### Relational analysis result of IS_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2843625, upper bound: 1.2828543
time: 7.25 seconds

## Relational analysis of IS_B2_B1_B2_A2

### Relational analysis result of IS_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2843625, upper bound: 1.2848952
time: 9.93 seconds

## BFS IS instance: IS_B2_B2_A1

### Backsubstitution after applying IS history:
0: -9.0066681, -5.5779705, -9.0238228, -5.5697422, -2.7776756, 2.8344641
1: -6.5625634, -3.9660497, -6.5725770, -3.9602938, -2.2656803, 2.2506685
2: 8.3391047, 10.9219522, 8.3258553, 10.9275665, -2.3123231, 2.3039749
3: -6.0989108, -2.9019959, -6.1187601, -2.8909538, -2.9815626, 2.9232106
4: -11.8234091, -7.9955416, -11.8287811, -7.9853377, -3.0698557, 3.0657663
5: -13.6544857, -10.1856394, -13.6605110, -10.1830397, -2.6192341, 2.6109815
6: -15.6420631, -12.3342628, -15.6547489, -12.3252039, -2.3933682, 2.4152932
7: -5.5268369, -2.0648127, -5.5500355, -2.0485888, -3.2549567, 3.2561030
8: -1.9538317, 0.3790965, -1.9596810, 0.3828821, -2.1020827, 2.1189590
9: -7.3032122, -4.0151958, -7.3081818, -4.0086079, -2.7719531, 2.7693024

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4639
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_B2_B2_A1_A1

### Relational analysis result of IS_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2848944, upper bound: 1.2811468
time: 10.16 seconds

## Relational analysis of IS_B2_B2_A1_A2

### Relational analysis result of IS_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2848944, upper bound: 1.2831837
time: 7.22 seconds

## BFS IS instance: IS_B2_B2_A2

### Backsubstitution after applying IS history:
0: -9.0258865, -5.5622640, -9.0258703, -5.5622663, -2.8044605, 2.8492870
1: -6.5765247, -3.9591043, -6.5765171, -3.9591107, -2.2791290, 2.2624803
2: 8.3243217, 10.9320316, 8.3243303, 10.9319973, -2.3189449, 2.3110256
3: -6.1232653, -2.8826222, -6.1232524, -2.8826299, -3.0058126, 2.9287271
4: -11.8333721, -7.9824538, -11.8333588, -7.9824634, -3.0799098, 3.0807850
5: -13.6636410, -10.1825562, -13.6636267, -10.1825571, -2.6284866, 2.6156754
6: -15.6556501, -12.3172064, -15.6556311, -12.3172073, -2.4118605, 2.4214487
7: -5.5685844, -2.0476770, -5.5685830, -2.0476897, -3.2870517, 3.2920747
8: -1.9611869, 0.3840785, -1.9611802, 0.3840694, -2.1174483, 2.1225672
9: -7.3109112, -4.0054560, -7.3108974, -4.0054712, -2.7839098, 2.7824240

Time for backsubstitution: 14.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 6191
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of IS_B2_B2_A2_A1

### Relational analysis result of IS_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714652, upper bound: 1.2714654
time: 6.99 seconds

## Relational analysis of IS_B2_B2_A2_A2

### Relational analysis result of IS_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714653, upper bound: 1.2848994
time: 5.97 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 27.74 seconds
IS_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 2, lower bound: -1.2714617, upper bound: 1.2767624
IS_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 2, lower bound: -1.2714617, upper bound: 1.2788172
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 2, lower bound: -1.2693883, upper bound: 1.2805221
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 2, lower bound: -1.2714616, upper bound: 1.2805216
IS_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 2, lower bound: -1.2677201, upper bound: 1.2809039
IS_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 2, lower bound: -1.2697889, upper bound: 1.2809036
IS_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 2, lower bound: -1.2693883, upper bound: 1.2809039
IS_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 2, lower bound: -1.2714617, upper bound: 1.2809058
IS_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 2, lower bound: -1.2806092, upper bound: 1.2848952
IS_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 2, lower bound: -1.2826447, upper bound: 1.2848951
IS_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 2, lower bound: -1.2843625, upper bound: 1.2828543
IS_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 2, lower bound: -1.2843625, upper bound: 1.2848952
IS_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 2, lower bound: -1.2848944, upper bound: 1.2811468
IS_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 2, lower bound: -1.2848944, upper bound: 1.2831837
IS_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 2, lower bound: -1.2714652, upper bound: 1.2714654
IS_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 27.74
Output dim: 2, lower bound: -1.2714653, upper bound: 1.2848994

## BFS IS instance: IS_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -8.9428778, -5.6028786, -8.9776573, -5.5895534, -2.7127733, 2.7365475
1: -6.5071588, -4.0189319, -6.5399046, -3.9793835, -2.1797652, 2.1729796
2: 8.3851347, 10.8809347, 8.3760567, 10.8723812, -2.2069345, 2.2204998
3: -6.0263062, -2.9739075, -6.0675087, -2.9237819, -2.8731384, 2.8595791
4: -11.7936230, -8.0334520, -11.8014832, -8.0175323, -3.0145774, 2.9997153
5: -13.6215010, -10.2001610, -13.6276646, -10.2005739, -2.5410137, 2.5486193
6: -15.5867290, -12.3542147, -15.6106606, -12.3447142, -2.3190899, 2.3511682
7: -5.4702330, -2.0955782, -5.5133214, -2.0781302, -3.1638265, 3.1801124
8: -1.9157486, 0.3485556, -1.9354787, 0.3701844, -2.0485463, 2.0564771
9: -7.2562523, -4.0696836, -7.2809319, -4.0472355, -2.6769047, 2.6881399

Time for backsubstitution: 14.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of IS_B1_A1_A1_A1_A1

### Relational analysis result of IS_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714617, upper bound: 1.2673519
time: 9.96 seconds

## Relational analysis of IS_B1_A1_A1_A1_A2

### Relational analysis result of IS_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714617, upper bound: 1.2767623
time: 11.69 seconds

## BFS IS instance: IS_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -8.9602242, -5.5830708, -8.9845676, -5.5890098, -2.7237196, 2.7647104
1: -6.5169287, -4.0100799, -6.5408249, -3.9767435, -2.1939554, 2.1814854
2: 8.3696098, 10.8905687, 8.3753395, 10.8760815, -2.2264915, 2.2293291
3: -6.0424175, -2.9643202, -6.0688376, -2.9203949, -2.8923774, 2.8670912
4: -11.7966909, -8.0258684, -11.8021193, -8.0166187, -3.0188246, 3.0083485
5: -13.6353111, -10.1936274, -13.6284180, -10.1977110, -2.5582218, 2.5548997
6: -15.6072063, -12.3294220, -15.6189651, -12.3441925, -2.3320441, 2.3755348
7: -5.4843006, -2.0844679, -5.5157790, -2.0738966, -3.1829948, 3.1912079
8: -1.9305072, 0.3647752, -1.9410129, 0.3706775, -2.0599823, 2.0794621
9: -7.2633429, -4.0606699, -7.2833004, -4.0459280, -2.6843748, 2.7000394

Time for backsubstitution: 14.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of IS_B1_A1_A1_A2_A1

### Relational analysis result of IS_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714617, upper bound: 1.2694056
time: 6.73 seconds

## Relational analysis of IS_B1_A1_A1_A2_A2

### Relational analysis result of IS_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714617, upper bound: 1.2788153
time: 12.32 seconds

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -8.9695253, -5.5865631, -8.9721308, -5.5826578, -2.7463675, 2.7445440
1: -6.5225110, -4.0090189, -6.5427990, -3.9810777, -2.1914244, 2.1871803
2: 8.3701258, 10.8950329, 8.3753300, 10.8727589, -2.2152071, 2.2393155
3: -6.0521011, -2.9507198, -6.0705709, -2.9191124, -2.9027562, 2.8678656
4: -11.8042736, -8.0194740, -11.8053789, -8.0156822, -3.0242071, 3.0177026
5: -13.6315403, -10.1939363, -13.6299706, -10.2032251, -2.5476685, 2.5559063
6: -15.6093426, -12.3365860, -15.6024408, -12.3372889, -2.3504200, 2.3484426
7: -5.5146341, -2.0738358, -5.5291781, -2.0818634, -3.1942658, 3.2176590
8: -1.9292479, 0.3540521, -1.9309421, 0.3708282, -2.0697417, 2.0541668
9: -7.2664948, -4.0584354, -7.2810564, -4.0455379, -2.6894765, 2.7008667

Time for backsubstitution: 14.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 6191
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of IS_B1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2693883, upper bound: 1.2711091
time: 7.30 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2693883, upper bound: 1.2805221
time: 11.01 seconds

## BFS IS instance: IS_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -8.9764385, -5.5860195, -8.9893417, -5.5641789, -2.7730074, 2.7554336
1: -6.5234156, -4.0062866, -6.5522747, -3.9726126, -2.1999974, 2.2010150
2: 8.3694506, 10.8987408, 8.3600483, 10.8822403, -2.2237210, 2.2530012
3: -6.0534062, -2.9472847, -6.0859060, -2.9093244, -2.9100780, 2.8866072
4: -11.8049126, -8.0186110, -11.8081064, -8.0081587, -3.0326796, 3.0215158
5: -13.6322985, -10.1910620, -13.6431398, -10.1968164, -2.5536051, 2.5724707
6: -15.6176720, -12.3360615, -15.6227036, -12.3140459, -2.3674388, 2.3608136
7: -5.5170650, -2.0695975, -5.5429692, -2.0709438, -3.2052474, 3.2365680
8: -1.9347978, 0.3545051, -1.9455290, 0.3871474, -2.0902710, 2.0654860
9: -7.2688642, -4.0571585, -7.2879772, -4.0366392, -2.7012177, 2.7081633

Time for backsubstitution: 14.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 6191
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of IS_B1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714616, upper bound: 1.2711088
time: 24.69 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714616, upper bound: 1.2805238
time: 22.78 seconds

## BFS IS instance: IS_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -9.0051250, -5.5776420, -8.9649220, -5.5967593, -2.7677555, 2.7423635
1: -6.5668116, -3.9698639, -6.5438499, -3.9864526, -2.2216153, 2.2236068
2: 8.3364305, 10.9009361, 8.3856659, 10.8660011, -2.2401304, 2.2516012
3: -6.1071773, -2.9052391, -6.0716724, -2.9365487, -2.8857327, 2.9305058
4: -11.8179846, -7.9905210, -11.7974739, -8.0193529, -3.0342493, 3.0351858
5: -13.6470232, -10.1870098, -13.6244469, -10.2037230, -2.5675240, 2.5636024
6: -15.6327877, -12.3307810, -15.5947933, -12.3524294, -2.3623295, 2.3559303
7: -5.5392714, -2.0622201, -5.4954853, -2.0940137, -3.2069769, 3.2047272
8: -1.9479337, 0.3807569, -1.9271455, 0.3729243, -2.0879431, 2.0785670
9: -7.2959275, -4.0146780, -7.2764816, -4.0367899, -2.7419615, 2.7365766

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 6191
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of IS_B1_A2_B1_B1_A1

### Relational analysis result of IS_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2677201, upper bound: 1.2714654
time: 11.23 seconds

## Relational analysis of IS_B1_A2_B1_B1_A2

### Relational analysis result of IS_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2677201, upper bound: 1.2809035
time: 11.90 seconds

## BFS IS instance: IS_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -9.0120468, -5.5770912, -8.9821110, -5.5782676, -2.7943668, 2.7530088
1: -6.5677319, -3.9672174, -6.5533319, -3.9779336, -2.2302904, 2.2373865
2: 8.3357143, 10.9046364, 8.3704166, 10.8754816, -2.2487283, 2.2630365
3: -6.1084714, -2.9018445, -6.0871372, -2.9265952, -2.8929567, 2.9416113
4: -11.8186226, -7.9896064, -11.8002110, -8.0118752, -3.0407786, 3.0390701
5: -13.6477842, -10.1841402, -13.6376419, -10.1973162, -2.5734367, 2.5801706
6: -15.6410942, -12.3302574, -15.6150055, -12.3291941, -2.3825183, 2.3683224
7: -5.5417156, -2.0579824, -5.5093136, -2.0830956, -3.2180200, 3.2236614
8: -1.9534678, 0.3812485, -1.9418359, 0.3891568, -2.1078105, 2.0897808
9: -7.2983036, -4.0133882, -7.2834215, -4.0279112, -2.7537012, 2.7438860

Time for backsubstitution: 14.66 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.3431549072265625
rel_dist={2: [-1.2849390101996399, 1.2849362165636915]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4639
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 6191
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1648064, upper bound: 1.1577658
time: 8.39 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688669, upper bound: 1.1688656
time: 5.54 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 14.18 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 14.18
Output dim: 2, lower bound: -1.1648064, upper bound: 1.1577658
IS_A2, status: Status.UNKNOWN, split count: 1, time: 14.18
Output dim: 2, lower bound: -1.1688669, upper bound: 1.1688656

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.0258865, -5.5622592, -9.0258932, -5.5622578, -2.7596579, 2.7251797
1: -6.5765324, -3.9591095, -6.5765352, -3.9591041, -2.2075744, 2.2108514
2: 8.3243237, 10.9320021, 8.3243179, 10.9320288, -2.2642121, 2.2531319
3: -6.1232662, -2.8826249, -6.1232753, -2.8826141, -2.9204884, 2.9423375
4: -11.8333654, -7.9824467, -11.8333740, -7.9824452, -2.9800320, 2.9852409
5: -13.6636353, -10.1825542, -13.6636448, -10.1825542, -2.4983912, 2.5127039
6: -15.6556358, -12.3172054, -15.6556473, -12.3171988, -2.3277702, 2.3129444
7: -5.5686011, -2.0476840, -5.5686092, -2.0476754, -3.2501774, 3.2462883
8: -1.9611835, 0.3840857, -1.9611893, 0.3840866, -2.0765114, 2.0609422
9: -7.3109016, -4.0054460, -7.3109131, -4.0054426, -2.7002239, 2.7123060

Time for backsubstitution: 14.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6191
type: B, layer: 1, pos: 6191
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6191

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688667, upper bound: 1.1684422
time: 6.18 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688667, upper bound: 1.1688654
time: 6.35 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 27.74 seconds
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 27.74
Output dim: 2, lower bound: -1.1688667, upper bound: 1.1684422
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 27.74
Output dim: 2, lower bound: -1.1688667, upper bound: 1.1688654

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -8.9886036, -5.5786982, -9.0120907, -5.5641479, -2.7186575, 2.6914306
1: -6.5282736, -3.9994304, -6.5585651, -3.9610291, -2.1524825, 2.1548803
2: 8.3598528, 10.9217129, 8.3297939, 10.9281368, -2.2127390, 2.2159693
3: -6.0636644, -2.9367185, -6.0927744, -2.8848262, -2.8564620, 2.8550034
4: -11.8151150, -8.0142670, -11.8309860, -7.9935980, -2.9479561, 2.9488010
5: -13.6450043, -10.1898537, -13.6593447, -10.1855888, -2.4744072, 2.4954662
6: -15.6315403, -12.3309956, -15.6488914, -12.3194504, -2.2977209, 2.2924590
7: -5.5253348, -2.0600867, -5.5589800, -2.0532870, -3.1980386, 3.2103419
8: -1.9408922, 0.3561544, -1.9568024, 0.3757172, -2.0397110, 2.0259309
9: -7.2787361, -4.0523582, -7.3072491, -4.0275269, -2.6376195, 2.6620765

Time for backsubstitution: 14.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 6111

## Relational analysis of IS_A2_A1_A1

### Relational analysis result of IS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688682, upper bound: 1.1561982
time: 10.37 seconds

## Relational analysis of IS_A2_A1_A2

### Relational analysis result of IS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688660, upper bound: 1.1684414
time: 5.81 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -9.0258732, -5.5622635, -9.0258865, -5.5622549, -2.7569723, 2.7126307
1: -6.5765219, -3.9591112, -6.5765285, -3.9591043, -2.1910357, 2.2108340
2: 8.3243294, 10.9320002, 8.3243217, 10.9320297, -2.2374563, 2.2514145
3: -6.1232567, -2.8826246, -6.1232700, -2.8826163, -2.8599691, 2.9305983
4: -11.8333616, -7.9824634, -11.8333721, -7.9824529, -2.9774084, 2.9735079
5: -13.6636276, -10.1825552, -13.6636410, -10.1825523, -2.5000515, 2.5119991
6: -15.6556339, -12.3172016, -15.6556473, -12.3171949, -2.3256855, 2.3129401
7: -5.5685940, -2.0476890, -5.5686054, -2.0476775, -3.2347441, 3.2400584
8: -1.9611812, 0.3840694, -1.9611878, 0.3840771, -2.0706987, 2.0565906
9: -7.3109002, -4.0054684, -7.3109107, -4.0054545, -2.7073889, 2.7092834

Time for backsubstitution: 15.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 6111

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1675576, upper bound: 1.1688672
time: 5.21 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688660, upper bound: 1.1688647
time: 6.59 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 27.09 seconds
IS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 27.09
Output dim: 2, lower bound: -1.1688682, upper bound: 1.1561982
IS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 27.09
Output dim: 2, lower bound: -1.1688660, upper bound: 1.1684414
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 27.09
Output dim: 2, lower bound: -1.1675576, upper bound: 1.1688672
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 27.09
Output dim: 2, lower bound: -1.1688660, upper bound: 1.1688647

## BFS IS instance: IS_A2_A1_A1

### Backsubstitution after applying IS history:
0: -8.9694948, -5.5944099, -9.0096474, -5.5730472, -2.6904621, 2.6730890
1: -6.5139165, -4.0063586, -6.5539174, -3.9624171, -2.1349897, 2.1422765
2: 8.3742714, 10.9116440, 8.3316364, 10.9228754, -2.1894212, 2.2002890
3: -6.0393186, -2.9561229, -6.0874238, -2.8947573, -2.8228283, 2.8303990
4: -11.8051472, -8.0272884, -11.8255301, -7.9970231, -2.9348783, 2.9258242
5: -13.6358128, -10.1929359, -13.6556282, -10.1861658, -2.4643745, 2.4889293
6: -15.6180477, -12.3480673, -15.6478329, -12.3289881, -2.2709174, 2.2729106
7: -5.4835715, -2.0771976, -5.5368633, -2.0543554, -3.1552739, 3.1708035
8: -1.9334793, 0.3511448, -1.9550104, 0.3742809, -2.0261068, 2.0153184
9: -7.2710423, -4.0622311, -7.3040180, -4.0312457, -2.6256018, 2.6481104

Time for backsubstitution: 14.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A2_A1_A1_A1

### Relational analysis result of IS_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688632, upper bound: 1.1656023
time: 18.32 seconds

## Relational analysis of IS_A2_A1_A1_A2

### Relational analysis result of IS_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688632, upper bound: 1.1671181
time: 8.34 seconds

## BFS IS instance: IS_A2_A1_A2

### Backsubstitution after applying IS history:
0: -8.9885998, -5.5787039, -9.0120888, -5.5641518, -2.7186432, 2.6881609
1: -6.5282674, -3.9994316, -6.5585604, -3.9610295, -2.1491542, 2.1548259
2: 8.3598566, 10.9217072, 8.3297958, 10.9281359, -2.1977777, 2.2151017
3: -6.0636568, -2.9367275, -6.0927696, -2.8848307, -2.8564477, 2.8361144
4: -11.8151102, -8.0142717, -11.8309841, -7.9935999, -2.9448605, 2.9436145
5: -13.6449966, -10.1898537, -13.6593437, -10.1855888, -2.4735203, 2.4938860
6: -15.6315413, -12.3310051, -15.6488905, -12.3194571, -2.2915759, 2.2786393
7: -5.5253115, -2.0600872, -5.5589662, -2.0532873, -3.1869574, 3.2103219
8: -1.9408894, 0.3561511, -1.9568009, 0.3757167, -2.0411863, 2.0185938
9: -7.2787313, -4.0523629, -7.3072457, -4.0275326, -2.6378980, 2.6620684

Time for backsubstitution: 14.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_A2_A1_A2_B1

### Relational analysis result of IS_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1673441, upper bound: 1.1684385
time: 11.58 seconds

## Relational analysis of IS_A2_A1_A2_B2

### Relational analysis result of IS_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688632, upper bound: 1.1684389
time: 6.38 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -9.0234261, -5.5711699, -9.0066652, -5.5779724, -2.7386589, 2.6843753
1: -6.5718241, -3.9605219, -6.5625620, -3.9660504, -2.1783800, 2.1938806
2: 8.3261499, 10.9267197, 8.3391075, 10.9219475, -2.2218542, 2.2281675
3: -6.1178889, -2.8925543, -6.0989070, -2.9019988, -2.8357553, 2.8945012
4: -11.8279037, -7.9858961, -11.8234062, -7.9955425, -2.9544382, 2.9604015
5: -13.6599121, -10.1831322, -13.6544828, -10.1856403, -2.4935341, 2.5019431
6: -15.6545763, -12.3267355, -15.6420593, -12.3342628, -2.3061128, 2.2846153
7: -5.5464826, -2.0487616, -5.5268331, -2.0648155, -3.1951342, 3.1972575
8: -1.9593973, 0.3826332, -1.9538293, 0.3790951, -2.0600820, 2.0428362
9: -7.3076639, -4.0092077, -7.3032079, -4.0152011, -2.6936064, 2.6972332

Time for backsubstitution: 15.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1660358, upper bound: 1.1688628
time: 7.28 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1675549, upper bound: 1.1688645
time: 5.90 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -9.0258732, -5.5622692, -9.0258846, -5.5622668, -2.7537031, 2.7126174
1: -6.5765166, -3.9591122, -6.5765219, -3.9591053, -2.1910000, 2.2074676
2: 8.3243313, 10.9319983, 8.3243237, 10.9320240, -2.2281761, 2.2350695
3: -6.1232524, -2.8826308, -6.1232624, -2.8826246, -2.8410807, 2.9200611
4: -11.8333578, -7.9824634, -11.8333693, -7.9824572, -2.9695854, 2.9682589
5: -13.6636238, -10.1825562, -13.6636362, -10.1825542, -2.4985476, 2.5111113
6: -15.6556339, -12.3172064, -15.6556454, -12.3172035, -2.3118901, 2.3044438
7: -5.5685806, -2.0476894, -5.5685816, -2.0476806, -3.2347231, 3.2289772
8: -1.9611783, 0.3840690, -1.9611859, 0.3840761, -2.0633588, 2.0580883
9: -7.3108969, -4.0054712, -7.3109074, -4.0054588, -2.7073789, 2.7095623

Time for backsubstitution: 14.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1673419, upper bound: 1.1688644
time: 8.37 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688632, upper bound: 1.1688620
time: 5.67 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 29.20 seconds
IS_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 29.20
Output dim: 2, lower bound: -1.1688632, upper bound: 1.1656023
IS_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 29.20
Output dim: 2, lower bound: -1.1688632, upper bound: 1.1671181
IS_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.20
Output dim: 2, lower bound: -1.1673441, upper bound: 1.1684385
IS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.20
Output dim: 2, lower bound: -1.1688632, upper bound: 1.1684389
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 29.20
Output dim: 2, lower bound: -1.1660358, upper bound: 1.1688628
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 29.20
Output dim: 2, lower bound: -1.1675549, upper bound: 1.1688645
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 29.20
Output dim: 2, lower bound: -1.1673419, upper bound: 1.1688644
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 29.20
Output dim: 2, lower bound: -1.1688632, upper bound: 1.1688620

## BFS IS instance: IS_A2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -8.9550142, -5.5955644, -9.0011806, -5.5737314, -2.6748562, 2.6630425
1: -6.5120282, -4.0120850, -6.5527864, -3.9656498, -2.1293054, 2.1345019
2: 8.3756466, 10.9038877, 8.3325081, 10.9183350, -2.1827602, 2.1908765
3: -6.0365453, -2.9632716, -6.0858397, -2.8989050, -2.8162589, 2.8218513
4: -11.8038187, -8.0291262, -11.8247433, -7.9981484, -2.9323282, 2.9230194
5: -13.6342125, -10.1989613, -13.6546917, -10.1896744, -2.4585228, 2.4809704
6: -15.6005650, -12.3491678, -15.6376476, -12.3296347, -2.2519236, 2.2610598
7: -5.4785137, -2.0860755, -5.5339022, -2.0595448, -3.1446066, 3.1586905
8: -1.9218359, 0.3501945, -1.9482279, 0.3736773, -2.0135417, 2.0074439
9: -7.2661524, -4.0648675, -7.3011131, -4.0328131, -2.6182475, 2.6412759

Time for backsubstitution: 14.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 6219

## Relational analysis of IS_A2_A1_A1_A1_B1

### Relational analysis result of IS_A2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1672372, upper bound: 1.1655995
time: 8.87 seconds

## Relational analysis of IS_A2_A1_A1_A1_B2

### Relational analysis result of IS_A2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688602, upper bound: 1.1655996
time: 6.32 seconds

## BFS IS instance: IS_A2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -8.9723568, -5.5757656, -9.0096340, -5.5730476, -2.6847115, 2.6929245
1: -6.5218010, -4.0032344, -6.5539160, -3.9624250, -2.1441946, 2.1430104
2: 8.3601360, 10.9135246, 8.3316412, 10.9228659, -2.1961746, 2.1993821
3: -6.0526538, -2.9537387, -6.0874205, -2.8947630, -2.8362474, 2.8290334
4: -11.8068790, -8.0215359, -11.8255272, -7.9970264, -2.9367642, 2.9317799
5: -13.6480083, -10.1924419, -13.6556282, -10.1861687, -2.4764395, 2.4870310
6: -15.6210432, -12.3243723, -15.6478109, -12.3289909, -2.2635322, 2.2811868
7: -5.4925294, -2.0749638, -5.5368595, -2.0543633, -3.1646395, 3.1699691
8: -1.9365973, 0.3664098, -1.9550018, 0.3742805, -2.0242162, 2.0317121
9: -7.2732754, -4.0558877, -7.3040113, -4.0312486, -2.6257906, 2.6536756

Time for backsubstitution: 14.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 6219

## Relational analysis of IS_A2_A1_A1_A2_B1

### Relational analysis result of IS_A2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1672394, upper bound: 1.1671150
time: 12.70 seconds

## Relational analysis of IS_A2_A1_A1_A2_B2

### Relational analysis result of IS_A2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688602, upper bound: 1.1671152
time: 7.37 seconds

## BFS IS instance: IS_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -8.9801311, -5.5793762, -8.9976120, -5.5653138, -2.7085485, 2.6726742
1: -6.5271587, -4.0027828, -6.5566258, -3.9665384, -2.1415877, 2.1489577
2: 8.3606758, 10.9171610, 8.3312893, 10.9203815, -2.1885071, 2.2082384
3: -6.0620522, -2.9409213, -6.0900383, -2.8918905, -2.8479939, 2.8294935
4: -11.8143234, -8.0153322, -11.8296518, -7.9955273, -2.9419861, 2.9411211
5: -13.6440687, -10.1933756, -13.6577377, -10.1915913, -2.4656296, 2.4880023
6: -15.6213226, -12.3316545, -15.6314726, -12.3205643, -2.2774975, 2.2597218
7: -5.5223494, -2.0652788, -5.5539055, -2.0621605, -3.1749573, 3.1994939
8: -1.9340878, 0.3555946, -1.9452024, 0.3746810, -2.0332928, 2.0060296
9: -7.2758317, -4.0539107, -7.3022871, -4.0302114, -2.6310220, 2.6547470

Time for backsubstitution: 15.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of IS_A2_A1_A2_B1_A1

### Relational analysis result of IS_A2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1673391, upper bound: 1.1668142
time: 6.77 seconds

## Relational analysis of IS_A2_A1_A2_B1_A2

### Relational analysis result of IS_A2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1673391, upper bound: 1.1684357
time: 5.38 seconds

## BFS IS instance: IS_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -8.9885855, -5.5787015, -9.0148621, -5.5455322, -2.7362399, 2.6825523
1: -6.5282640, -3.9994376, -6.5664425, -3.9580555, -2.1500945, 2.1639454
2: 8.3598576, 10.9217014, 8.3153515, 10.9300156, -2.1969275, 2.2229826
3: -6.0636535, -2.9367323, -6.1061344, -2.8823023, -2.8552113, 2.8495574
4: -11.8151083, -8.0142736, -11.8327122, -7.9878702, -2.9507179, 2.9454970
5: -13.6449966, -10.1898565, -13.6715088, -10.1851597, -2.4713726, 2.5059299
6: -15.6315193, -12.3310032, -15.6517525, -12.2957802, -2.2940617, 2.2708077
7: -5.5253072, -2.0600948, -5.5678539, -2.0511036, -3.1861944, 3.2195563
8: -1.9408789, 0.3561502, -1.9599085, 0.3910537, -2.0497859, 2.0166855
9: -7.2787261, -4.0523658, -7.3093958, -4.0211625, -2.6434636, 2.6622567

Time for backsubstitution: 14.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of IS_A2_A1_A2_B2_A1

### Relational analysis result of IS_A2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688602, upper bound: 1.1668138
time: 14.70 seconds

## Relational analysis of IS_A2_A1_A2_B2_A2

### Relational analysis result of IS_A2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688602, upper bound: 1.1684358
time: 6.62 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -9.0149479, -5.5718508, -8.9921675, -5.5791378, -2.7286119, 2.6687803
1: -6.5706959, -3.9637673, -6.5606375, -3.9715936, -2.1707625, 2.1881132
2: 8.3270226, 10.9221821, 8.3405800, 10.9142017, -2.2124877, 2.2204976
3: -6.1163011, -2.8967068, -6.0961637, -2.9090750, -2.8271837, 2.8871779
4: -11.8271151, -7.9870186, -11.8220739, -7.9974813, -2.9514542, 2.9578481
5: -13.6589737, -10.1866426, -13.6528683, -10.1916466, -2.4856224, 2.4960785
6: -15.6443882, -12.3273869, -15.6246281, -12.3353653, -2.2920954, 2.2656620
7: -5.5435042, -2.0539575, -5.5217552, -2.0737000, -3.1830988, 3.1865129
8: -1.9526181, 0.3820281, -1.9422235, 0.3780594, -2.0521083, 2.0302463
9: -7.3047523, -4.0107718, -7.2982492, -4.0178804, -2.6866913, 2.6899009

Time for backsubstitution: 14.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of IS_A2_A2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1660307, upper bound: 1.1672363
time: 8.78 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1660307, upper bound: 1.1688595
time: 10.27 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -9.0234146, -5.5711708, -9.0093918, -5.5593443, -2.7537632, 2.6784291
1: -6.5718222, -3.9605279, -6.5704594, -3.9630601, -2.1794043, 2.2030282
2: 8.3261499, 10.9267120, 8.3246670, 10.9238300, -2.2209940, 2.2332261
3: -6.1178865, -2.8925586, -6.1122904, -2.8993001, -2.8341837, 2.8994670
4: -11.8279047, -7.9858956, -11.8251429, -7.9898820, -2.9579854, 2.9622998
5: -13.6599092, -10.1831360, -13.6666660, -10.1852198, -2.4913406, 2.5138729
6: -15.6545534, -12.3267345, -15.6448517, -12.3105831, -2.3086066, 2.2768717
7: -5.5464783, -2.0487697, -5.5357399, -2.0626426, -3.1944122, 3.2066064
8: -1.9593859, 0.3826337, -1.9570332, 0.3943281, -2.0678875, 2.0408096
9: -7.3076582, -4.0092096, -7.3053799, -4.0088539, -2.6991425, 2.6974273

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of IS_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1675518, upper bound: 1.1672360
time: 9.20 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1675518, upper bound: 1.1688614
time: 5.53 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -9.0173931, -5.5629501, -9.0113878, -5.5634255, -2.7436609, 2.6970339
1: -6.5753870, -3.9623566, -6.5745912, -3.9646385, -2.1833982, 2.2016897
2: 8.3252058, 10.9274597, 8.3258200, 10.9242725, -2.2188110, 2.2273645
3: -6.1216640, -2.8867834, -6.1205320, -2.8896945, -2.8325248, 2.9127240
4: -11.8325691, -7.9835835, -11.8320322, -7.9843817, -2.9666042, 2.9656932
5: -13.6626873, -10.1860676, -13.6620283, -10.1885624, -2.4906454, 2.5052443
6: -15.6454420, -12.3178558, -15.6382236, -12.3183174, -2.2978656, 2.2854953
7: -5.5656066, -2.0528867, -5.5634966, -2.0565634, -3.2226734, 3.2182083
8: -1.9543972, 0.3834639, -1.9495912, 0.3830357, -2.0553803, 2.0455022
9: -7.3079820, -4.0070353, -7.3059368, -4.0081353, -2.7004743, 2.7022243

Time for backsubstitution: 14.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of IS_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1673391, upper bound: 1.1672363
time: 8.83 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1673412, upper bound: 1.1688614
time: 9.18 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -9.0258560, -5.5622692, -9.0286436, -5.5436344, -2.7629700, 2.7067022
1: -6.5765152, -3.9591191, -6.5844049, -3.9560950, -2.1920419, 2.2165709
2: 8.3243322, 10.9319897, 8.3098249, 10.9339066, -2.2273169, 2.2401106
3: -6.1232500, -2.8826361, -6.1366549, -2.8799465, -2.8394518, 2.9250307
4: -11.8333578, -7.9824667, -11.8351002, -7.9767656, -2.9731355, 2.9701352
5: -13.6636238, -10.1825581, -13.6758099, -10.1821356, -2.4963512, 2.5213623
6: -15.6556101, -12.3172073, -15.6584778, -12.2935314, -2.3143814, 2.2966950
7: -5.5685768, -2.0476985, -5.5774693, -2.0455008, -3.2340121, 3.2382927
8: -1.9611683, 0.3840690, -1.9643316, 0.3993049, -2.0702724, 2.0562282
9: -7.3108902, -4.0054750, -7.3130684, -3.9991193, -2.7129078, 2.7097411

Time for backsubstitution: 15.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of IS_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688602, upper bound: 1.1672357
time: 6.40 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688602, upper bound: 1.1688589
time: 6.82 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 28.59 seconds
IS_A2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 28.59
Output dim: 2, lower bound: -1.1672372, upper bound: 1.1655995
IS_A2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 28.59
Output dim: 2, lower bound: -1.1688602, upper bound: 1.1655996
IS_A2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 28.59
Output dim: 2, lower bound: -1.1672394, upper bound: 1.1671150
IS_A2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 28.59
Output dim: 2, lower bound: -1.1688602, upper bound: 1.1671152
IS_A2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.59
Output dim: 2, lower bound: -1.1673391, upper bound: 1.1668142
IS_A2_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.59
Output dim: 2, lower bound: -1.1673391, upper bound: 1.1684357
IS_A2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.59
Output dim: 2, lower bound: -1.1688602, upper bound: 1.1668138
IS_A2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.59
Output dim: 2, lower bound: -1.1688602, upper bound: 1.1684358
IS_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.59
Output dim: 2, lower bound: -1.1660307, upper bound: 1.1672363
IS_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.59
Output dim: 2, lower bound: -1.1660307, upper bound: 1.1688595
IS_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.59
Output dim: 2, lower bound: -1.1675518, upper bound: 1.1672360
IS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.59
Output dim: 2, lower bound: -1.1675518, upper bound: 1.1688614
IS_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.59
Output dim: 2, lower bound: -1.1673391, upper bound: 1.1672363
IS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.59
Output dim: 2, lower bound: -1.1673412, upper bound: 1.1688614
IS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.59
Output dim: 2, lower bound: -1.1688602, upper bound: 1.1672357
IS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.59
Output dim: 2, lower bound: -1.1688602, upper bound: 1.1688589

## BFS IS instance: IS_A2_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -8.9413090, -5.5968933, -8.9675608, -5.5986524, -2.6271858, 2.6210604
1: -6.5077510, -4.0169344, -6.5341663, -3.9771662, -2.1019249, 2.0955381
2: 8.3794794, 10.8967838, 8.3585424, 10.9039516, -2.1577716, 2.1470380
3: -6.0312910, -2.9717827, -6.0561013, -2.9236829, -2.7900801, 2.7798543
4: -11.7904968, -8.0341024, -11.8002987, -8.0308762, -2.8854771, 2.8932447
5: -13.6310215, -10.2064867, -13.6370354, -10.2033081, -2.4384604, 2.4481435
6: -15.5838184, -12.3518486, -15.6010571, -12.3639107, -2.1937239, 2.2168932
7: -5.4726200, -2.0935836, -5.5077534, -2.0785942, -3.1099501, 3.1102996
8: -1.9041300, 0.3465858, -1.9117951, 0.3447113, -1.9570689, 1.9583020
9: -7.2622185, -4.0674462, -7.2877216, -4.0463805, -2.5930772, 2.6185222

Time for backsubstitution: 15.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_A1_A1_A1_B1_B1

### Relational analysis result of IS_A2_A1_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1562877, upper bound: 1.1617063
time: 5.51 seconds

## Relational analysis of IS_A2_A1_A1_A1_B1_B2

### Relational analysis result of IS_A2_A1_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1562877, upper bound: 1.1656007
time: 6.04 seconds

## BFS IS instance: IS_A2_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -8.9549961, -5.5955648, -9.0011539, -5.5737329, -2.6748333, 2.6480579
1: -6.5120249, -4.0120883, -6.5527792, -3.9656558, -2.1271915, 2.1395524
2: 8.3756523, 10.9038754, 8.3325138, 10.9183216, -2.1713061, 2.1859431
3: -6.0365419, -2.9632845, -6.0858321, -2.8989255, -2.8146529, 2.8256507
4: -11.8037996, -8.0291309, -11.8247137, -7.9981546, -2.9258432, 2.9111378
5: -13.6342087, -10.1989698, -13.6546879, -10.1896925, -2.4562817, 2.4863124
6: -15.6005564, -12.3491697, -15.6376333, -12.3296366, -2.2318654, 2.2344904
7: -5.4785099, -2.0860794, -5.5338941, -2.0595534, -3.1427894, 3.1630039
8: -1.9218197, 0.3501906, -1.9482017, 0.3736744, -2.0041347, 1.9858179
9: -7.2661510, -4.0648699, -7.3011074, -4.0328178, -2.6194944, 2.6407442

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 6191
type: A, layer: 1, pos: 6219

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_A2_A1_A1_A1_B2_A1

### Relational analysis result of IS_A2_A1_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1658794, upper bound: 1.1655359
time: 12.09 seconds

## Relational analysis of IS_A2_A1_A1_A1_B2_A2

### Relational analysis result of IS_A2_A1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688533, upper bound: 1.1655932
time: 14.18 seconds

## BFS IS instance: IS_A2_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -8.9586067, -5.5770969, -8.9759493, -5.5979605, -2.6370344, 2.6508555
1: -6.5175443, -4.0080824, -6.5353236, -3.9739301, -2.1168690, 2.1038141
2: 8.3639345, 10.9064064, 8.3576488, 10.9084711, -2.1711035, 2.1554224
3: -6.0473785, -2.9623518, -6.0577340, -2.9196396, -2.8099413, 2.7871094
4: -11.7935591, -8.0264111, -11.8010893, -8.0296555, -2.8899946, 2.9021134
5: -13.6448517, -10.1999817, -13.6379976, -10.1998177, -2.4563780, 2.4541841
6: -15.6043262, -12.3270140, -15.6112194, -12.3632050, -2.2053607, 2.2366147
7: -5.4866734, -2.0824625, -5.5107841, -2.0734506, -3.1300426, 3.1216869
8: -1.9189267, 0.3628950, -1.9185805, 0.3454175, -1.9679065, 1.9825349
9: -7.2693558, -4.0584607, -7.2905927, -4.0447922, -2.6007366, 2.6309280

Time for backsubstitution: 14.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 6191
type: A, layer: 1, pos: 6219

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_A2_A1_A1_A2_B1_A1

### Relational analysis result of IS_A2_A1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1642582, upper bound: 1.1670518
time: 6.32 seconds

## Relational analysis of IS_A2_A1_A1_A2_B1_A2

### Relational analysis result of IS_A2_A1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1672303, upper bound: 1.1671111
time: 19.99 seconds

## BFS IS instance: IS_A2_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -8.9723434, -5.5757661, -9.0096092, -5.5730491, -2.6846876, 2.6779404
1: -6.5217972, -4.0032382, -6.5539093, -3.9624295, -2.1420813, 2.1480613
2: 8.3601389, 10.9135151, 8.3316460, 10.9228516, -2.1840439, 2.1944089
3: -6.0526485, -2.9537506, -6.0874128, -2.8947852, -2.8346424, 2.8328342
4: -11.8068638, -8.0215416, -11.8255005, -7.9970336, -2.9303665, 2.9179018
5: -13.6480045, -10.1924553, -13.6556225, -10.1861868, -2.4741936, 2.4923739
6: -15.6210327, -12.3243771, -15.6477947, -12.3289890, -2.2435298, 2.2509508
7: -5.4925261, -2.0749698, -5.5368514, -2.0543706, -3.1628222, 3.1742830
8: -1.9365826, 0.3664069, -1.9549770, 0.3742757, -2.0149150, 2.0053694
9: -7.2732720, -4.0558877, -7.3040056, -4.0312510, -2.6270370, 2.6531463

Time for backsubstitution: 14.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 6191
type: A, layer: 1, pos: 6219

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4616

## Relational analysis of IS_A2_A1_A1_A2_B2_A1

### Relational analysis result of IS_A2_A1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1658794, upper bound: 1.1670514
time: 12.07 seconds

## Relational analysis of IS_A2_A1_A1_A2_B2_A2

### Relational analysis result of IS_A2_A1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688533, upper bound: 1.1671088
time: 5.76 seconds

## BFS IS instance: IS_A2_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8.9462032, -5.6042919, -8.9838820, -5.5666728, -2.6674309, 2.6259608
1: -6.5087032, -4.0142727, -6.5523314, -3.9714086, -2.1017547, 2.1230195
2: 8.3863087, 10.9027233, 8.3353872, 10.9132538, -2.1446114, 2.1831224
3: -6.0321493, -2.9650402, -6.0848422, -2.9004016, -2.8069925, 2.8029184
4: -11.7898932, -8.0477648, -11.8163338, -8.0006599, -2.9122772, 2.8925622
5: -13.6264277, -10.2069407, -13.6545162, -10.1991539, -2.4327164, 2.4679875
6: -15.5843773, -12.3659582, -15.6148148, -12.3232460, -2.2337182, 2.2049918
7: -5.4960742, -2.0843120, -5.5479851, -2.0697043, -3.1269007, 3.1650901
8: -1.8975458, 0.3263702, -1.9276133, 0.3711510, -1.9822841, 1.9516845
9: -7.2624207, -4.0674219, -7.2982664, -4.0327802, -2.6084509, 2.6296406

Time for backsubstitution: 14.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 6191
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 6219

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_A1_A2_B1_A1_B1

### Relational analysis result of IS_A2_A1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1562338, upper bound: 1.1630637
time: 22.31 seconds

## Relational analysis of IS_A2_A1_A2_B1_A1_B2

### Relational analysis result of IS_A2_A1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1562324, upper bound: 1.1630648
time: 16.52 seconds

## BFS IS instance: IS_A2_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -8.9801035, -5.5793772, -8.9975939, -5.5653133, -2.6935616, 2.6726508
1: -6.5271521, -4.0027890, -6.5566211, -3.9665422, -2.1466403, 2.1468453
2: 8.3606815, 10.9171448, 8.3312950, 10.9203739, -2.1778154, 2.1978812
3: -6.0620441, -2.9409437, -6.0900326, -2.8919020, -2.8517952, 2.8278880
4: -11.8142967, -8.0153360, -11.8296356, -7.9955301, -2.9304166, 2.9291151
5: -13.6440639, -10.1933947, -13.6577339, -10.1916037, -2.4709797, 2.4857607
6: -15.6213055, -12.3316574, -15.6314678, -12.3205643, -2.2472413, 2.2447803
7: -5.5223408, -2.0652874, -5.5538998, -2.0621648, -3.1792698, 3.1976786
8: -1.9340620, 0.3555918, -1.9451876, 0.3746777, -2.0100698, 2.0004058
9: -7.2758250, -4.0539155, -7.3022847, -4.0302138, -2.6304898, 2.6559944

Time for backsubstitution: 14.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 6219

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_A1_A2_B1_A2_B1

### Relational analysis result of IS_A2_A1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1562324, upper bound: 1.1645228
time: 16.39 seconds

## Relational analysis of IS_A2_A1_A2_B1_A2_B2

### Relational analysis result of IS_A2_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1562324, upper bound: 1.1684367
time: 14.41 seconds

## BFS IS instance: IS_A2_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.9545507, -5.6036143, -9.0010881, -5.5468912, -2.6952796, 2.6358662
1: -6.5097780, -4.0109153, -6.5621743, -3.9629192, -2.1102209, 2.1380606
2: 8.3855333, 10.9072495, 8.3194218, 10.9228716, -2.1528535, 2.1979153
3: -6.0338011, -2.9609976, -6.1009169, -2.8909230, -2.8142657, 2.8228054
4: -11.7906780, -8.0466328, -11.8193932, -7.9928875, -2.9211164, 2.8970251
5: -13.6273737, -10.2034378, -13.6683168, -10.1927338, -2.4384279, 2.4859171
6: -15.5945215, -12.3652487, -15.6351223, -12.2984257, -2.2500868, 2.2161212
7: -5.4991059, -2.0791688, -5.5619717, -2.0586360, -3.1382723, 3.1852069
8: -1.9043074, 0.3270297, -1.9423513, 0.3876176, -1.9982271, 1.9624829
9: -7.2652841, -4.0659008, -7.3053818, -4.0237312, -2.6208906, 2.6372352

Time for backsubstitution: 14.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 6219

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 4616

## Relational analysis of IS_A2_A1_A2_B2_A1_B1

### Relational analysis result of IS_A2_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1687966, upper bound: 1.1638332
time: 6.35 seconds

## Relational analysis of IS_A2_A1_A2_B2_A1_B2

### Relational analysis result of IS_A2_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688537, upper bound: 1.1668074
time: 7.34 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 28.79 seconds
IS_A2_A1_A1_A1_B1_B1, status: Status.VERIFIED, split count: 6, time: 28.79
Output dim: 2, lower bound: -1.1562877, upper bound: 1.1617063
IS_A2_A1_A1_A1_B1_B2, status: Status.VERIFIED, split count: 6, time: 28.79
Output dim: 2, lower bound: -1.1562877, upper bound: 1.1656007
IS_A2_A1_A1_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 28.79
Output dim: 2, lower bound: -1.1658794, upper bound: 1.1655359
IS_A2_A1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 28.79
Output dim: 2, lower bound: -1.1688533, upper bound: 1.1655932
IS_A2_A1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 28.79
Output dim: 2, lower bound: -1.1642582, upper bound: 1.1670518
IS_A2_A1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 28.79
Output dim: 2, lower bound: -1.1672303, upper bound: 1.1671111
IS_A2_A1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 28.79
Output dim: 2, lower bound: -1.1658794, upper bound: 1.1670514
IS_A2_A1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 28.79
Output dim: 2, lower bound: -1.1688533, upper bound: 1.1671088
IS_A2_A1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 28.79
Output dim: 2, lower bound: -1.1562338, upper bound: 1.1630637
IS_A2_A1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 28.79
Output dim: 2, lower bound: -1.1562324, upper bound: 1.1630648
IS_A2_A1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 28.79
Output dim: 2, lower bound: -1.1562324, upper bound: 1.1645228
IS_A2_A1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 28.79
Output dim: 2, lower bound: -1.1562324, upper bound: 1.1684367
IS_A2_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 28.79
Output dim: 2, lower bound: -1.1687966, upper bound: 1.1638332
IS_A2_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 28.79
Output dim: 2, lower bound: -1.1688537, upper bound: 1.1668074
IS_A2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.79
Output dim: 2, lower bound: -1.1688602, upper bound: 1.1684358
IS_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.79
Output dim: 2, lower bound: -1.1660307, upper bound: 1.1672363
IS_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.79
Output dim: 2, lower bound: -1.1660307, upper bound: 1.1688595
IS_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.79
Output dim: 2, lower bound: -1.1675518, upper bound: 1.1672360
IS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.79
Output dim: 2, lower bound: -1.1675518, upper bound: 1.1688614
IS_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 28.79
Output dim: 2, lower bound: -1.1673391, upper bound: 1.1672363
IS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 28.79
Output dim: 2, lower bound: -1.1673412, upper bound: 1.1688614
IS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 28.79
Output dim: 2, lower bound: -1.1688602, upper bound: 1.1672357
IS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 28.79
Output dim: 2, lower bound: -1.1688602, upper bound: 1.1688589
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.2642369270324707
rel_dist={2: [-1.168894797061638, 1.1688945587998152]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2431.88 seconds
