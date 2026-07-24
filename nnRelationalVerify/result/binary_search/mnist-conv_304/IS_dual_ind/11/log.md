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
execution time: IAR + LP analysis = 14.59 + 33.60 = 48.18 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3551.82 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.5009913444519043
rel_dist={2: [-1.4846913114101667, 1.4846931118005955]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.2642369270324707
rel_dist={2: [-1.1688970347914687, 1.1688946419177597]}

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
Binary search time: 223.89 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 3327.93 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5704881, upper bound: 1.5533021
time: 6.17 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5739634, upper bound: 1.5739622
time: 5.96 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 12.37 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 12.37
Output dim: 2, lower bound: -1.5704881, upper bound: 1.5533021
IS_A2, status: Status.UNKNOWN, split count: 1, time: 12.37
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

Time for backsubstitution: 14.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5533021, upper bound: 1.5533021
time: 11.85 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5533021, upper bound: 1.5533024
time: 6.10 seconds

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

Time for backsubstitution: 15.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6219

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5700695, upper bound: 1.5739548
time: 9.58 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5739559, upper bound: 1.5739547
time: 5.17 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 40.10 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 40.10
Output dim: 2, lower bound: -1.5533021, upper bound: 1.5533021
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 40.10
Output dim: 2, lower bound: -1.5533021, upper bound: 1.5533024
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 40.10
Output dim: 2, lower bound: -1.5700695, upper bound: 1.5739548
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 40.10
Output dim: 2, lower bound: -1.5739559, upper bound: 1.5739547

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -8.9986153, -5.5798907, -8.9986153, -5.5798907, -3.0483179, 3.0483174
1: -6.5597734, -3.9739642, -6.5597734, -3.9739642, -2.4481225, 2.4481220
2: 8.3694963, 10.8838148, 8.3694963, 10.8838148, -2.4906607, 2.4906607
3: -6.0988617, -2.9099305, -6.0988617, -2.9099305, -3.1889312, 3.1889312
4: -11.8087769, -8.0043459, -11.8087769, -8.0043459, -3.3678002, 3.3678002
5: -13.6352262, -10.1946335, -13.6352262, -10.1946335, -2.9118128, 2.9118128
6: -15.6257658, -12.3342810, -15.6257658, -12.3342810, -2.6953721, 2.6953721
7: -5.5424242, -2.0679922, -5.5424242, -2.0679922, -3.4134989, 3.4134984
8: -1.9461651, 0.3789444, -1.9461651, 0.3789444, -2.2718625, 2.2718630
9: -7.2891226, -4.0242567, -7.2891226, -4.0242567, -2.9618626, 2.9618630

Time for backsubstitution: 14.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4666

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5489336, upper bound: 1.5532902
time: 6.23 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5532973, upper bound: 1.5532898
time: 7.92 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -8.9986153, -5.5798907, -9.0258865, -5.5622592, -3.0707283, 3.0700912
1: -6.5597734, -3.9739642, -6.5765324, -3.9591095, -2.4685416, 2.4699011
2: 8.3694963, 10.8838148, 8.3243237, 10.9320021, -2.5412941, 2.5292375
3: -6.0988617, -2.9099305, -6.1232662, -2.8826249, -3.2162368, 3.2133358
4: -11.8087769, -8.0043459, -11.8333654, -7.9824467, -3.3904176, 3.3956451
5: -13.6352262, -10.1946335, -13.6636353, -10.1825542, -2.9244699, 2.9488969
6: -15.6257658, -12.3342810, -15.6556358, -12.3172054, -2.7132864, 2.7201443
7: -5.5424242, -2.0679922, -5.5686011, -2.0476840, -3.4420519, 3.4439769
8: -1.9461651, 0.3789444, -1.9611835, 0.3840857, -2.2781067, 2.2865133
9: -7.2891226, -4.0242567, -7.3109016, -4.0054460, -2.9825773, 2.9864941

Time for backsubstitution: 15.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4666

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5489336, upper bound: 1.5532911
time: 5.92 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5532973, upper bound: 1.5532899
time: 4.94 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.0194511, -5.5629148, -8.9921799, -5.5871868, -3.0979290, 3.0582404
1: -6.5745749, -3.9613929, -6.5579348, -3.9706326, -2.4661016, 2.4591916
2: 8.3262253, 10.9286804, 8.3504744, 10.9176264, -2.5582829, 2.5318623
3: -6.1208568, -2.8866811, -6.0933685, -2.9072466, -3.2136102, 3.2066875
4: -11.8271742, -7.9847493, -11.8089371, -8.0150414, -3.3743906, 3.3910499
5: -13.6621580, -10.1860695, -13.6460495, -10.1961765, -2.9478884, 2.9519696
6: -15.6478806, -12.3184118, -15.6190987, -12.3514471, -2.7166042, 2.6963742
7: -5.5658336, -2.0512276, -5.5424857, -2.0667806, -3.4434018, 3.4280672
8: -1.9530029, 0.3824978, -1.9246650, 0.3552017, -2.2643261, 2.2434592
9: -7.3090177, -4.0066490, -7.2974515, -4.0189428, -2.9755850, 2.9857802

Time for backsubstitution: 14.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5700625, upper bound: 1.5700979
time: 7.51 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5700625, upper bound: 1.5739477
time: 9.95 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.0258751, -5.5622625, -9.0258760, -5.5622540, -3.1334028, 3.0834446
1: -6.5765305, -3.9591103, -6.5765324, -3.9591053, -2.4894748, 2.5031693
2: 8.3243256, 10.9320011, 8.3243179, 10.9320412, -2.5733099, 2.5659785
3: -6.1232648, -2.8826280, -6.1232753, -2.8826294, -3.2406354, 3.2406473
4: -11.8333597, -7.9824486, -11.8333530, -7.9824486, -3.4140453, 3.4101458
5: -13.6636353, -10.1825562, -13.6636486, -10.1825714, -2.9652271, 2.9874444
6: -15.6556339, -12.3172007, -15.6556473, -12.3171959, -2.7475131, 2.7183223
7: -5.5686011, -2.0476851, -5.5686088, -2.0476751, -3.4718037, 3.4758072
8: -1.9611778, 0.3840842, -1.9611721, 0.3840842, -2.3056068, 2.2764139
9: -7.3109016, -4.0054469, -7.3109174, -4.0054412, -3.0001364, 3.0070004

Time for backsubstitution: 14.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5739491, upper bound: 1.5700974
time: 13.20 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5739491, upper bound: 1.5739478
time: 4.94 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 32.98 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 32.98
Output dim: 2, lower bound: -1.5489336, upper bound: 1.5532902
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 32.98
Output dim: 2, lower bound: -1.5532973, upper bound: 1.5532898
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 32.98
Output dim: 2, lower bound: -1.5489336, upper bound: 1.5532911
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 32.98
Output dim: 2, lower bound: -1.5532973, upper bound: 1.5532899
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 32.98
Output dim: 2, lower bound: -1.5700625, upper bound: 1.5700979
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 32.98
Output dim: 2, lower bound: -1.5700625, upper bound: 1.5739477
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 32.98
Output dim: 2, lower bound: -1.5739491, upper bound: 1.5700974
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 32.98
Output dim: 2, lower bound: -1.5739491, upper bound: 1.5739478

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.9955444, -5.5823212, -8.9978008, -5.5800238, -3.0451736, 3.0445471
1: -6.5446234, -3.9989862, -6.5594130, -3.9805615, -2.4253826, 2.4224439
2: 8.3841248, 10.8751965, 8.3729496, 10.8835783, -2.4752903, 2.4774091
3: -6.0884390, -2.9145379, -6.0959702, -2.9099598, -3.1784792, 3.1814322
4: -11.8051128, -8.0099001, -11.8085146, -8.0058498, -3.3625765, 3.3619032
5: -13.6317520, -10.1964617, -13.6344032, -10.1947355, -2.9082432, 2.9086714
6: -15.6130857, -12.3421011, -15.6221485, -12.3345261, -2.6823463, 2.6827369
7: -5.5273466, -2.0914774, -5.5420227, -2.0746405, -3.3891597, 3.3895907
8: -1.9350753, 0.3739777, -1.9453793, 0.3774781, -2.2578835, 2.2662950
9: -7.2819738, -4.0306025, -7.2871284, -4.0246305, -2.9537563, 2.9532027

Time for backsubstitution: 14.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5450722, upper bound: 1.5532906
time: 14.96 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5489263, upper bound: 1.5532909
time: 8.78 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.9986115, -5.5798931, -8.9986134, -5.5798903, -3.0489702, 3.0475521
1: -6.5597725, -3.9739885, -6.5597739, -3.9739680, -2.4481125, 2.4269545
2: 8.3695068, 10.8838139, 8.3694983, 10.8838139, -2.4822989, 2.4906585
3: -6.0988512, -2.9099312, -6.0988612, -2.9099312, -3.1889200, 3.1889300
4: -11.8087759, -8.0043488, -11.8087740, -8.0043449, -3.3677983, 3.3650417
5: -13.6352215, -10.1946354, -13.6352253, -10.1946344, -2.9116344, 2.9118118
6: -15.6257534, -12.3342829, -15.6257610, -12.3342829, -2.6849766, 2.6947532
7: -5.5424228, -2.0680251, -5.5424232, -2.0679967, -3.4121013, 3.3951750
8: -1.9461627, 0.3789396, -1.9461656, 0.3789439, -2.2718601, 2.2680111
9: -7.2891178, -4.0242567, -7.2891216, -4.0242558, -2.9568377, 2.9618602

Time for backsubstitution: 15.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5494353, upper bound: 1.5532927
time: 8.89 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5532907, upper bound: 1.5532905
time: 9.74 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.9955444, -5.5823212, -9.0250454, -5.5629396, -3.0668573, 3.0662589
1: -6.5446234, -3.9989862, -6.5760403, -3.9657111, -2.4457860, 2.4440181
2: 8.3841248, 10.8751965, 8.3280401, 10.9317045, -2.5258231, 2.5157778
3: -6.0884390, -2.9145379, -6.1200886, -2.8826556, -3.2057834, 3.2055507
4: -11.8051128, -8.0099001, -11.8329592, -7.9840059, -3.3851414, 3.3895826
5: -13.6317520, -10.1964617, -13.6625528, -10.1826649, -2.9208908, 2.9454842
6: -15.6130857, -12.3421011, -15.6520081, -12.3180866, -2.6996374, 2.7074723
7: -5.5273466, -2.0914774, -5.5680809, -2.0543931, -3.4176445, 3.4199338
8: -1.9350753, 0.3739777, -1.9603357, 0.3826027, -2.2640929, 2.2808495
9: -7.2819738, -4.0306025, -7.3088312, -4.0059080, -2.9743738, 2.9777613

Time for backsubstitution: 15.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6219

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5622109, upper bound: 1.5532829
time: 6.37 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5661042, upper bound: 1.5532828
time: 10.80 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.9986115, -5.5798931, -9.0258837, -5.5622597, -3.0714197, 3.0693254
1: -6.5597725, -3.9739885, -6.5765324, -3.9591134, -2.4685316, 2.4487433
2: 8.3695068, 10.8838139, 8.3243265, 10.9320021, -2.5329323, 2.5286944
3: -6.0988512, -2.9099312, -6.1232653, -2.8826246, -3.2123404, 3.2133341
4: -11.8087759, -8.0043488, -11.8333645, -7.9824467, -3.3904157, 3.3928847
5: -13.6352215, -10.1946354, -13.6636314, -10.1825533, -2.9242916, 2.9488964
6: -15.6257534, -12.3342829, -15.6556349, -12.3172045, -2.7029481, 2.7195249
7: -5.5424228, -2.0680251, -5.5686011, -2.0476880, -3.4406567, 3.4256377
8: -1.9461627, 0.3789396, -1.9611835, 0.3840857, -2.2781034, 2.2826605
9: -7.2891178, -4.0242567, -7.3109026, -4.0054464, -2.9775519, 2.9864922

Time for backsubstitution: 15.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6219

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5665754, upper bound: 1.5532848
time: 11.71 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5704678, upper bound: 1.5532827
time: 9.09 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.0049725, -5.5640783, -8.9890032, -5.5874472, -3.0829487, 3.0535645
1: -6.5726299, -3.9669268, -6.5574970, -3.9718604, -2.4626632, 2.4523370
2: 8.3277302, 10.9209347, 8.3508139, 10.9159193, -2.5544095, 2.5232096
3: -6.1181259, -2.8937180, -6.0927534, -2.9087782, -3.2093477, 3.1990354
4: -11.8258410, -7.9867105, -11.8086376, -8.0154982, -3.3725653, 3.3886418
5: -13.6605349, -10.1920719, -13.6456881, -10.1974974, -2.9445019, 2.9447360
6: -15.6304626, -12.3195391, -15.6152554, -12.3517141, -2.6981044, 2.6911414
7: -5.5607309, -2.0600998, -5.5413380, -2.0687280, -3.4358783, 3.4180431
8: -1.9414058, 0.3814306, -1.9221029, 0.3549347, -2.2520189, 2.2398319
9: -7.3040466, -4.0093298, -7.2963624, -4.0195403, -2.9696331, 2.9808078

Time for backsubstitution: 14.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5493664, upper bound: 1.5666231
time: 12.16 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5493665, upper bound: 1.5700986
time: 5.77 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.0222082, -5.5442882, -8.9921741, -5.5871873, -3.0973845, 3.0780964
1: -6.5824513, -3.9583797, -6.5579319, -3.9706349, -2.4752541, 2.4612317
2: 8.3117256, 10.9305620, 8.3504744, 10.9176254, -2.5706272, 2.5331485
3: -6.1342373, -2.8840482, -6.0933671, -2.9072499, -3.2269874, 3.2093189
4: -11.8289032, -7.9790478, -11.8089371, -8.0150433, -3.3763943, 3.3968987
5: -13.6743288, -10.1856508, -13.6460505, -10.1961775, -2.9599543, 2.9514527
6: -15.6507282, -12.2947388, -15.6190853, -12.3514471, -2.7147727, 2.7091353
7: -5.5747199, -2.0490317, -5.5424824, -2.0667851, -3.4527502, 3.4291711
8: -1.9561610, 0.3977375, -1.9246593, 0.3552008, -2.2657804, 2.2596288
9: -7.3111849, -4.0003104, -7.2974482, -4.0189428, -2.9772253, 2.9913964

Time for backsubstitution: 14.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5493664, upper bound: 1.5704743
time: 33.72 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5493665, upper bound: 1.5739489
time: 6.70 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.0113811, -5.5634232, -9.0226765, -5.5625076, -3.1184158, 3.0787258
1: -6.5745983, -3.9646440, -6.5761070, -3.9603279, -2.4860630, 2.4963014
2: 8.3258209, 10.9242496, 8.3246469, 10.9303274, -2.5694079, 2.5573678
3: -6.1205368, -2.8896954, -6.1226807, -2.8841968, -3.2363400, 3.2329853
4: -11.8320208, -7.9843750, -11.8330564, -7.9828682, -3.4122505, 3.4077706
5: -13.6620197, -10.1885681, -13.6632957, -10.1838951, -2.9618435, 2.9802184
6: -15.6382122, -12.3183117, -15.6518040, -12.3174381, -2.7289639, 2.7130976
7: -5.5635128, -2.0565691, -5.5674901, -2.0496356, -3.4642758, 3.4657969
8: -1.9495845, 0.3830442, -1.9586139, 0.3838568, -2.2932749, 2.2728219
9: -7.3059301, -4.0081229, -7.3098183, -4.0060287, -2.9941826, 3.0020103

Time for backsubstitution: 14.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5532884, upper bound: 1.5666216
time: 10.35 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5532884, upper bound: 1.5701005
time: 14.64 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.0286369, -5.5436344, -9.0258675, -5.5622540, -3.1328526, 3.1032925
1: -6.5844126, -3.9561009, -6.5765314, -3.9591074, -2.4986362, 2.5051913
2: 8.3098297, 10.9338808, 8.3243179, 10.9320364, -2.5836449, 2.5672963
3: -6.1366568, -2.8799496, -6.1232748, -2.8826320, -3.2540247, 3.2433252
4: -11.8350868, -7.9767599, -11.8333530, -7.9824467, -3.4160471, 3.4159789
5: -13.6758041, -10.1821384, -13.6636467, -10.1825714, -2.9772921, 2.9869308
6: -15.6584644, -12.2935295, -15.6556349, -12.3171978, -2.7457733, 2.7285099
7: -5.5774875, -2.0455079, -5.5686054, -2.0476789, -3.4811211, 3.4768958
8: -1.9643245, 0.3993134, -1.9611659, 0.3840847, -2.3070583, 2.2927575
9: -7.3130631, -3.9991045, -7.3109159, -4.0054421, -3.0017223, 3.0126019

Time for backsubstitution: 15.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5532884, upper bound: 1.5704725
time: 8.63 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5532884, upper bound: 1.5739508
time: 7.98 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 31.91 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.91
Output dim: 2, lower bound: -1.5450722, upper bound: 1.5532906
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.91
Output dim: 2, lower bound: -1.5489263, upper bound: 1.5532909
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.91
Output dim: 2, lower bound: -1.5494353, upper bound: 1.5532927
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.91
Output dim: 2, lower bound: -1.5532907, upper bound: 1.5532905
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.91
Output dim: 2, lower bound: -1.5622109, upper bound: 1.5532829
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.91
Output dim: 2, lower bound: -1.5661042, upper bound: 1.5532828
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.91
Output dim: 2, lower bound: -1.5665754, upper bound: 1.5532848
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.91
Output dim: 2, lower bound: -1.5704678, upper bound: 1.5532827
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.91
Output dim: 2, lower bound: -1.5493664, upper bound: 1.5666231
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.91
Output dim: 2, lower bound: -1.5493665, upper bound: 1.5700986
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.91
Output dim: 2, lower bound: -1.5493664, upper bound: 1.5704743
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.91
Output dim: 2, lower bound: -1.5493665, upper bound: 1.5739489
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.91
Output dim: 2, lower bound: -1.5532884, upper bound: 1.5666216
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.91
Output dim: 2, lower bound: -1.5532884, upper bound: 1.5701005
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.91
Output dim: 2, lower bound: -1.5532884, upper bound: 1.5704725
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.91
Output dim: 2, lower bound: -1.5532884, upper bound: 1.5739508

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.9923410, -5.5825725, -8.9832993, -5.5811691, -3.0403967, 3.0294681
1: -6.5441999, -4.0002141, -6.5574851, -3.9860973, -2.4184923, 2.4190276
2: 8.3844576, 10.8734856, 8.3744555, 10.8758364, -2.4666810, 2.4735484
3: -6.0878301, -2.9161067, -6.0931783, -2.9170299, -3.1708002, 3.1770716
4: -11.8048191, -8.0103178, -11.8071909, -8.0077686, -3.3601847, 3.3601131
5: -13.6314049, -10.1977854, -13.6328144, -10.2007351, -2.9010181, 2.9052877
6: -15.6092386, -12.3423405, -15.6047258, -12.3356171, -2.6771450, 2.6642413
7: -5.5262089, -2.0934403, -5.5368528, -2.0835242, -3.3791695, 3.3820024
8: -1.9325185, 0.3737507, -1.9337893, 0.3764338, -2.2542830, 2.2540026
9: -7.2808781, -4.0312061, -7.2821655, -4.0273695, -2.9487219, 2.9472718

Time for backsubstitution: 15.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6111

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5450703, upper bound: 1.5504440
time: 9.53 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5450707, upper bound: 1.5532887
time: 9.48 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.9955349, -5.5823236, -9.0005207, -5.5626822, -3.0632553, 3.0434332
1: -6.5446234, -3.9989882, -6.5669575, -3.9775679, -2.4273567, 2.4311097
2: 8.3841248, 10.8751907, 8.3591166, 10.8853264, -2.4763875, 2.4910467
3: -6.0884390, -2.9145408, -6.1086140, -2.9071455, -3.1812935, 3.1940732
4: -11.8051138, -8.0099010, -11.8099232, -8.0002890, -3.3682818, 3.3635335
5: -13.6317520, -10.1964655, -13.6459875, -10.1943321, -2.9077101, 2.9201012
6: -15.6130753, -12.3421001, -15.6249752, -12.3123779, -2.6992698, 2.6807837
7: -5.5273457, -2.0914817, -5.5506706, -2.0725925, -3.3901076, 3.3986616
8: -1.9350686, 0.3739786, -1.9484062, 0.3926644, -2.2741537, 2.2674880
9: -7.2819719, -4.0306044, -7.2891107, -4.0184989, -2.9591455, 2.9546261

Time for backsubstitution: 14.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6111

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5489222, upper bound: 1.5504435
time: 9.47 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5489248, upper bound: 1.5532910
time: 9.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.9954128, -5.5801458, -8.9841156, -5.5810404, -3.0441933, 3.0324945
1: -6.5593476, -3.9752138, -6.5578465, -3.9795015, -2.4412217, 2.4235425
2: 8.3698387, 10.8821030, 8.3710041, 10.8760738, -2.4736876, 2.4867985
3: -6.0982404, -2.9114974, -6.0960674, -2.9170008, -3.1812396, 3.1845701
4: -11.8084793, -8.0047703, -11.8074512, -8.0062647, -3.3654070, 3.3632498
5: -13.6348743, -10.1959591, -13.6336327, -10.2006369, -2.9044104, 2.9084263
6: -15.6219101, -12.3345213, -15.6083469, -12.3353739, -2.6797776, 2.6762581
7: -5.5412860, -2.0699837, -5.5372543, -2.0768769, -3.4021111, 3.3875837
8: -1.9436064, 0.3787117, -1.9345756, 0.3779001, -2.2682610, 2.2557178
9: -7.2880192, -4.0248585, -7.2841568, -4.0269928, -2.9518099, 2.9559288

Time for backsubstitution: 14.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6111

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5494334, upper bound: 1.5504437
time: 10.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5494338, upper bound: 1.5532896
time: 6.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.9986038, -5.5798936, -9.0013313, -5.5625496, -3.0670567, 3.0464706
1: -6.5597715, -3.9739914, -6.5673170, -3.9709754, -2.4500895, 2.4356291
2: 8.3695097, 10.8838120, 8.3556728, 10.8855591, -2.4833903, 2.5042994
3: -6.0988474, -2.9099321, -6.1115236, -2.9071162, -3.1917312, 3.2015915
4: -11.8087740, -8.0043507, -11.8101845, -7.9987755, -3.3735089, 3.3666716
5: -13.6352215, -10.1946363, -13.6468143, -10.1942329, -2.9111032, 2.9232435
6: -15.6257429, -12.3342838, -15.6285868, -12.3121357, -2.7009807, 2.6927891
7: -5.5424204, -2.0680292, -5.5510721, -2.0659535, -3.4130402, 3.4042459
8: -1.9461584, 0.3789396, -1.9491935, 0.3941307, -2.2872486, 2.2692060
9: -7.2891145, -4.0242591, -7.2911024, -4.0181231, -2.9622316, 2.9632874

Time for backsubstitution: 14.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6111

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5532866, upper bound: 1.5504443
time: 6.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5532891, upper bound: 1.5532893
time: 13.92 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -8.9888668, -5.5829234, -8.9913330, -5.5873222, -3.0321045, 3.0280647
1: -6.5427198, -4.0013418, -6.5575666, -3.9772487, -2.4216714, 2.4062414
2: 8.3858175, 10.8718863, 8.3539295, 10.9173260, -2.5046263, 2.4769654
3: -6.0859957, -2.9189830, -6.0904937, -2.9072900, -3.1787057, 3.1715107
4: -11.7989435, -8.0120544, -11.8086586, -8.0165281, -3.3455529, 3.3625937
5: -13.6303148, -10.1999884, -13.6452312, -10.1962795, -2.9026737, 2.9170036
6: -15.6051655, -12.3432579, -15.6154375, -12.3517046, -2.6565304, 2.6658649
7: -5.5246210, -2.0950711, -5.5420580, -2.0734715, -3.3886762, 3.3779135
8: -1.9268289, 0.3723903, -1.9238610, 0.3537364, -2.2172604, 2.2301316
9: -7.2800550, -4.0317621, -7.2954388, -4.0193210, -2.9515753, 2.9564667

Time for backsubstitution: 14.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5622032, upper bound: 1.5494206
time: 7.17 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5622032, upper bound: 1.5532764
time: 5.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.9955368, -5.5823212, -9.0250216, -5.5629420, -3.0668449, 3.0541310
1: -6.5446215, -3.9989870, -6.5760341, -3.9657183, -2.4449248, 2.4503458
2: 8.3841286, 10.8751917, 8.3280487, 10.9316883, -2.5202184, 2.5108387
3: -6.0884385, -2.9145443, -6.1200800, -2.8826771, -3.2057614, 3.2055357
4: -11.8051071, -8.0099001, -11.8329325, -7.9840126, -3.3824062, 3.3814497
5: -13.6317530, -10.1964674, -13.6625471, -10.1826820, -2.9199772, 2.9521894
6: -15.6130810, -12.3421030, -15.6519928, -12.3180885, -2.6896815, 2.6877108
7: -5.5273447, -2.0914807, -5.5680728, -2.0544002, -3.4169011, 3.4253426
8: -1.9350705, 0.3739767, -1.9603105, 0.3825974, -2.2620108, 2.2633424
9: -7.2819734, -4.0306053, -7.3088236, -4.0059118, -2.9759402, 2.9775424

Time for backsubstitution: 15.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5660968, upper bound: 1.5494222
time: 7.13 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5660968, upper bound: 1.5532763
time: 13.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.9919462, -5.5805001, -8.9921589, -5.5871940, -3.0359249, 3.0310392
1: -6.5578680, -3.9763243, -6.5579271, -3.9706461, -2.4444160, 2.4107215
2: 8.3711929, 10.8805027, 8.3504839, 10.9175777, -2.5116353, 2.4884305
3: -6.0964041, -2.9143770, -6.0933518, -2.9072607, -3.1891434, 3.1789749
4: -11.8026047, -8.0065060, -11.8089190, -8.0150452, -3.3507624, 3.3657322
5: -13.6337919, -10.1981602, -13.6460333, -10.1961784, -2.9060707, 2.9201221
6: -15.6178493, -12.3354416, -15.6190710, -12.3514490, -2.6591511, 2.6779041
7: -5.5396895, -2.0716026, -5.5424709, -2.0668020, -3.4116383, 3.3834863
8: -1.9379072, 0.3773475, -1.9246545, 0.3551989, -2.2312193, 2.2318487
9: -7.2871933, -4.0254221, -7.2974300, -4.0189524, -2.9546437, 2.9651136

Time for backsubstitution: 15.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5665684, upper bound: 1.5494209
time: 5.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.5665684, upper bound: 1.5532767
time: 8.67 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.9986038, -5.5798912, -9.0258560, -5.5622649, -3.0714064, 3.0571980
1: -6.5597706, -3.9739904, -6.5765247, -3.9591193, -2.4676690, 2.4550714
2: 8.3695087, 10.8838120, 8.3243322, 10.9319878, -2.5259767, 2.5180039
3: -6.0988488, -2.9099369, -6.1232572, -2.8826456, -3.2116804, 3.2133203
4: -11.8087683, -8.0043526, -11.8333349, -7.9824548, -3.3861670, 3.3847528
5: -13.6352234, -10.1946383, -13.6636267, -10.1825724, -2.9233770, 2.9556007
6: -15.6257486, -12.3342848, -15.6556187, -12.3172054, -2.6920397, 2.6982546
7: -5.5424223, -2.0680256, -5.5685935, -2.0476959, -3.4399099, 3.4310455
8: -1.9461570, 0.3789387, -1.9611578, 0.3840814, -2.2714162, 2.2651505
9: -7.2891169, -4.0242591, -7.3108959, -4.0054512, -2.9791203, 2.9862719

Time for backsubstitution: 15.14 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=2.579909086227417
rel_dist={2: [-1.574020518019939, 1.5740203796460204]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2809087, upper bound: 1.2714660
time: 19.64 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2849001, upper bound: 1.2848987
time: 32.73 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 52.62 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 52.62
Output dim: 2, lower bound: -1.2809087, upper bound: 1.2714660
IS_A2, status: Status.UNKNOWN, split count: 1, time: 52.62
Output dim: 2, lower bound: -1.2849001, upper bound: 1.2848987

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -8.9986153, -5.5798907, -9.0141144, -5.5696034, -2.7963886, 2.7948146
1: -6.5597734, -3.9739642, -6.5716915, -3.9660313, -2.2454677, 2.2566376
2: 8.3694963, 10.8838148, 8.3341942, 10.9090824, -2.2806926, 2.2863641
3: -6.0988617, -2.9099305, -6.1129909, -2.8935089, -2.9739122, 2.9742837
4: -11.8087769, -8.0043459, -11.8232069, -7.9867201, -3.0611587, 3.0597548
5: -13.6352262, -10.1946335, -13.6509228, -10.1836557, -2.5792122, 2.5796609
6: -15.6257658, -12.3342810, -15.6420031, -12.3222532, -2.3998504, 2.3928728
7: -5.5424242, -2.0679922, -5.5602894, -2.0570686, -3.2616649, 3.2681718
8: -1.9461651, 0.3789444, -1.9549789, 0.3824530, -2.1077843, 2.1109905
9: -7.2891226, -4.0242567, -7.3010244, -4.0102215, -2.7575932, 2.7538757

Time for backsubstitution: 15.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714663, upper bound: 1.2714660
time: 10.82 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714663, upper bound: 1.2714662
time: 7.69 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -9.0258865, -5.5622592, -9.0258961, -5.5622540, -2.8530989, 2.8170185
1: -6.5765324, -3.9591095, -6.5765371, -3.9591014, -2.2782660, 2.2823491
2: 8.3243237, 10.9320021, 8.3243179, 10.9320374, -2.3431334, 2.3325751
3: -6.1232662, -2.8826249, -6.1232762, -2.8826139, -3.0041332, 3.0259595
4: -11.8333654, -7.9824467, -11.8333778, -7.9824438, -3.0885382, 3.0935001
5: -13.6636353, -10.1825542, -13.6636486, -10.1825542, -2.6154127, 2.6297126
6: -15.6556358, -12.3172054, -15.6556530, -12.3171959, -2.4359670, 2.4192305
7: -5.5686011, -2.0476840, -5.5686121, -2.0476735, -3.3057714, 3.3023171
8: -1.9611835, 0.3840857, -1.9611921, 0.3840876, -2.1352043, 2.1189003
9: -7.3109016, -4.0054460, -7.3109159, -4.0054383, -2.7748094, 2.7860365

Time for backsubstitution: 15.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714663, upper bound: 1.2809087
time: 20.35 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714664, upper bound: 1.2848999
time: 12.69 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 48.44 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 48.44
Output dim: 2, lower bound: -1.2714663, upper bound: 1.2714660
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 48.44
Output dim: 2, lower bound: -1.2714663, upper bound: 1.2714662
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 48.44
Output dim: 2, lower bound: -1.2714663, upper bound: 1.2809087
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 48.44
Output dim: 2, lower bound: -1.2714664, upper bound: 1.2848999

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -8.9986153, -5.5798907, -8.9986153, -5.5798907, -2.7728148, 2.7728143
1: -6.5597734, -3.9739642, -6.5597734, -3.9739642, -2.2360559, 2.2360568
2: 8.3694963, 10.8838148, 8.3694963, 10.8838148, -2.2539062, 2.2539065
3: -6.0988617, -2.9099305, -6.0988617, -2.9099305, -2.9536972, 2.9536977
4: -11.8087769, -8.0043459, -11.8087769, -8.0043459, -3.0430293, 3.0430298
5: -13.6352262, -10.1946335, -13.6352262, -10.1946335, -2.5626497, 2.5626502
6: -15.6257658, -12.3342810, -15.6257658, -12.3342810, -2.3765244, 2.3765249
7: -5.5424242, -2.0679922, -5.5424242, -2.0679922, -3.2467260, 3.2467265
8: -1.9461651, 0.3789444, -1.9461651, 0.3789444, -2.0979958, 2.0979958
9: -7.2891226, -4.0242567, -7.2891226, -4.0242567, -2.7406807, 2.7406812

Time for backsubstitution: 14.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4666

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2689706, upper bound: 1.2714596
time: 13.38 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714633, upper bound: 1.2714598
time: 27.43 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -8.9986153, -5.5798907, -9.0258865, -5.5622592, -2.7952251, 2.7945881
1: -6.5597734, -3.9739642, -6.5765324, -3.9591095, -2.2564759, 2.2578359
2: 8.3694963, 10.8838148, 8.3243237, 10.9320021, -2.2955427, 2.2924833
3: -6.0988617, -2.9099305, -6.1232662, -2.8826249, -2.9713764, 2.9823213
4: -11.8087769, -8.0043459, -11.8333654, -7.9824467, -3.0656466, 3.0708747
5: -13.6352262, -10.1946335, -13.6636353, -10.1825542, -2.5753069, 2.5997338
6: -15.6257658, -12.3342810, -15.6556358, -12.3172054, -2.3944387, 2.4012976
7: -5.5424242, -2.0679922, -5.5686011, -2.0476840, -3.2752810, 3.2772055
8: -1.9461651, 0.3789444, -1.9611835, 0.3840857, -2.1042395, 2.1126461
9: -7.2891226, -4.0242567, -7.3109016, -4.0054460, -2.7613964, 2.7653127

Time for backsubstitution: 14.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4666

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2689707, upper bound: 1.2714616
time: 15.75 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714634, upper bound: 1.2714604
time: 8.61 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -9.0258865, -5.5622592, -8.9986153, -5.5798907, -2.7945881, 2.7952251
1: -6.5765324, -3.9591095, -6.5597734, -3.9739642, -2.2578359, 2.2564764
2: 8.3243237, 10.9320021, 8.3694963, 10.8838148, -2.2924833, 2.2955427
3: -6.1232662, -2.8826249, -6.0988617, -2.9099305, -2.9823208, 2.9713764
4: -11.8333654, -7.9824467, -11.8087769, -8.0043459, -3.0708747, 3.0656466
5: -13.6636353, -10.1825542, -13.6352262, -10.1946335, -2.5997343, 2.5753069
6: -15.6556358, -12.3172054, -15.6257658, -12.3342810, -2.4012971, 2.3944383
7: -5.5686011, -2.0476840, -5.5424242, -2.0679922, -3.2772055, 3.2752800
8: -1.9611835, 0.3840857, -1.9461651, 0.3789444, -2.1126461, 2.1042395
9: -7.3109016, -4.0054460, -7.2891226, -4.0242567, -2.7653122, 2.7613964

Time for backsubstitution: 14.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714629, upper bound: 1.2788793
time: 17.64 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714629, upper bound: 1.2809037
time: 13.61 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.0258865, -5.5622592, -9.0258865, -5.5622592, -2.8530741, 2.8530741
1: -6.5765324, -3.9591095, -6.5765324, -3.9591095, -2.2823329, 2.2823331
2: 8.3243237, 10.9320021, 8.3243237, 10.9320021, -2.3325691, 2.3325689
3: -6.1232662, -2.8826249, -6.1232662, -2.8826249, -3.0259347, 3.0259352
4: -11.8333654, -7.9824467, -11.8333654, -7.9824467, -3.0885329, 3.0885334
5: -13.6636353, -10.1825542, -13.6636353, -10.1825542, -2.6296949, 2.6296945
6: -15.6556358, -12.3172054, -15.6556358, -12.3172054, -2.4359484, 2.4359488
7: -5.5686011, -2.0476840, -5.5686011, -2.0476840, -3.3023009, 3.3023014
8: -1.9611835, 0.3840857, -1.9611835, 0.3840857, -2.1351957, 2.1351953
9: -7.3109016, -4.0054460, -7.3109016, -4.0054460, -2.7748003, 2.7748003

Time for backsubstitution: 14.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714629, upper bound: 1.2826831
time: 17.16 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714629, upper bound: 1.2848962
time: 9.66 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 41.90 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 41.90
Output dim: 2, lower bound: -1.2689706, upper bound: 1.2714596
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 41.90
Output dim: 2, lower bound: -1.2714633, upper bound: 1.2714598
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 41.90
Output dim: 2, lower bound: -1.2689707, upper bound: 1.2714616
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 41.90
Output dim: 2, lower bound: -1.2714634, upper bound: 1.2714604
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 41.90
Output dim: 2, lower bound: -1.2714629, upper bound: 1.2788793
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 41.90
Output dim: 2, lower bound: -1.2714629, upper bound: 1.2809037
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 41.90
Output dim: 2, lower bound: -1.2714629, upper bound: 1.2826831
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 41.90
Output dim: 2, lower bound: -1.2714629, upper bound: 1.2848962

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -8.9955444, -5.5823212, -8.9972439, -5.5801153, -2.7695723, 2.7688184
1: -6.5446234, -3.9989862, -6.5591645, -3.9850848, -2.2081428, 2.2101157
2: 8.3841248, 10.8751965, 8.3753147, 10.8834152, -2.2383847, 2.2376056
3: -6.0884390, -2.9145379, -6.0939875, -2.9099813, -2.9432993, 2.9442191
4: -11.8051128, -8.0099001, -11.8083324, -8.0068827, -3.0367312, 3.0369825
5: -13.6317520, -10.1964617, -13.6338387, -10.1948042, -2.5589657, 2.5589566
6: -15.6130857, -12.3421011, -15.6196709, -12.3346958, -2.3632822, 2.3610778
7: -5.5273466, -2.0914774, -5.5417457, -2.0792022, -3.2169728, 3.2225513
8: -1.9350753, 0.3739777, -1.9448338, 0.3764710, -2.0830112, 2.0919919
9: -7.2819738, -4.0306025, -7.2857580, -4.0248866, -2.7322655, 2.7305846

Time for backsubstitution: 14.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2668943, upper bound: 1.2714600
time: 9.80 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2689667, upper bound: 1.2714597
time: 8.98 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -8.9986115, -5.5798931, -8.9986134, -5.5798912, -2.7727280, 2.7720475
1: -6.5597725, -3.9739885, -6.5597720, -3.9739749, -2.2360411, 2.2102551
2: 8.3695068, 10.8838139, 8.3695011, 10.8838139, -2.2429943, 2.2539001
3: -6.0988512, -2.9099312, -6.0988569, -2.9099312, -2.9453955, 2.9536920
4: -11.8087759, -8.0043488, -11.8087730, -8.0043468, -3.0430269, 3.0397849
5: -13.6352215, -10.1946354, -13.6352262, -10.1946344, -2.5624428, 2.5626478
6: -15.6257534, -12.3342829, -15.6257591, -12.3342819, -2.3636723, 2.3759012
7: -5.5424228, -2.0680251, -5.5424242, -2.0680058, -3.2453246, 3.2235618
8: -1.9461627, 0.3789396, -1.9461646, 0.3789425, -2.0979919, 2.0927753
9: -7.2891178, -4.0242567, -7.2891216, -4.0242558, -2.7347775, 2.7406764

Time for backsubstitution: 15.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2693869, upper bound: 1.2714599
time: 17.99 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714598, upper bound: 1.2714619
time: 10.44 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.9955444, -5.5823212, -9.0244732, -5.5634084, -2.7907524, 2.7904892
1: -6.5446234, -3.9989862, -6.5757046, -3.9702394, -2.2285376, 2.2315483
2: 8.3841248, 10.8751965, 8.3305902, 10.9314957, -2.2798433, 2.2758310
3: -6.0884390, -2.9145379, -6.1179066, -2.8826783, -2.9609756, 2.9725180
4: -11.8051128, -8.0099001, -11.8326778, -7.9850750, -3.0592604, 3.0645514
5: -13.6317520, -10.1964617, -13.6618118, -10.1827393, -2.5716095, 2.5955811
6: -15.6130857, -12.3421011, -15.6495180, -12.3186970, -2.3801451, 2.3857424
7: -5.5273466, -2.0914774, -5.5677199, -2.0589962, -3.2454123, 3.2528000
8: -1.9350753, 0.3739777, -1.9597468, 0.3815842, -2.0892029, 2.1064811
9: -7.2819738, -4.0306025, -7.3074079, -4.0062261, -2.7528162, 2.7550941

Time for backsubstitution: 15.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6219

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2763792, upper bound: 1.2714567
time: 8.76 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2784060, upper bound: 1.2714582
time: 12.96 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.9986115, -5.5798931, -9.0258808, -5.5622606, -2.7951736, 2.7938213
1: -6.5597725, -3.9739885, -6.5765305, -3.9591188, -2.2564621, 2.2320433
2: 8.3695068, 10.8838139, 8.3243294, 10.9320021, -2.2822704, 2.2880459
3: -6.0988512, -2.9099312, -6.1232610, -2.8826237, -2.9630728, 2.9823160
4: -11.8087759, -8.0043488, -11.8333645, -7.9824505, -3.0656433, 3.0676284
5: -13.6352215, -10.1946354, -13.6636333, -10.1825533, -2.5750971, 2.5997329
6: -15.6257534, -12.3342829, -15.6556330, -12.3172016, -2.3816438, 2.3958049
7: -5.5424228, -2.0680251, -5.5686007, -2.0476980, -3.2738791, 3.2540255
8: -1.9461627, 0.3789396, -1.9611816, 0.3840828, -2.1042352, 2.1074247
9: -7.2891178, -4.0242567, -7.3109007, -4.0054479, -2.7554932, 2.7653084

Time for backsubstitution: 15.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6219

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2788721, upper bound: 1.2714560
time: 10.09 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2808973, upper bound: 1.2714582
time: 9.52 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -8.9921589, -5.5871921, -8.9865875, -5.5809641, -2.7531719, 2.7510724
1: -6.5579267, -3.9706440, -6.5563087, -3.9781592, -2.2180777, 2.2301135
2: 8.3504820, 10.9175768, 8.3725452, 10.8778353, -2.2493787, 2.2719195
3: -6.0933533, -2.9072607, -6.0944881, -2.9179087, -2.9428091, 2.9460182
4: -11.8089209, -8.0150442, -11.7976265, -8.0082817, -3.0420828, 3.0210366
5: -13.6460323, -10.1961775, -13.6326313, -10.2010059, -2.5670815, 2.5558553
6: -15.6190720, -12.3514490, -15.6115017, -12.3364048, -2.3580432, 2.3428054
7: -5.5424709, -2.0667968, -5.5375013, -2.0744379, -3.2300491, 3.2423396
8: -1.9246545, 0.3551998, -1.9312534, 0.3760543, -2.0619726, 2.0512886
9: -7.2974319, -4.0189524, -7.2856522, -4.0263309, -2.7430964, 2.7367802

Time for backsubstitution: 14.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4666

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714562, upper bound: 1.2763791
time: 8.40 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714563, upper bound: 1.2788718
time: 6.67 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.0258579, -5.5622654, -8.9986010, -5.5798950, -2.7803221, 2.7952042
1: -6.5765252, -3.9591157, -6.5597696, -3.9739676, -2.2632074, 2.2547307
2: 8.3243313, 10.9319868, 8.3695011, 10.8838072, -2.2849913, 2.2852156
3: -6.1232595, -2.8826456, -6.0988574, -2.9099422, -2.9863648, 2.9700508
4: -11.8333359, -7.9824533, -11.8087616, -8.0043488, -3.0612545, 3.0582757
5: -13.6636286, -10.1825724, -13.6352215, -10.1946430, -2.6054220, 2.5734572
6: -15.6556225, -12.3172045, -15.6257563, -12.3342829, -2.3748930, 2.3810604
7: -5.5685935, -2.0476909, -5.5424213, -2.0679960, -3.2817926, 3.2737794
8: -1.9611592, 0.3840823, -1.9461532, 0.3789434, -2.0920496, 2.0998704
9: -7.3108959, -4.0054493, -7.2891212, -4.0242577, -2.7648726, 2.7627234

Time for backsubstitution: 14.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4666

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714562, upper bound: 1.2784062
time: 19.37 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2714563, upper bound: 1.2808972
time: 9.86 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.9921589, -5.5871921, -9.0142813, -5.5634165, -2.8126326, 2.8089180
1: -6.5579267, -3.9706440, -6.5729742, -3.9632068, -2.2427845, 2.2562172
2: 8.3504820, 10.9175768, 8.3277607, 10.9260025, -2.2894530, 2.3084469
3: -6.0933533, -2.9072607, -6.1189108, -2.8898945, -2.9864616, 3.0008516
4: -11.8089209, -8.0150442, -11.8221798, -7.9866529, -3.0596180, 3.0439081
5: -13.6460323, -10.1961775, -13.6609554, -10.1889029, -2.5975628, 2.6101851
6: -15.6190720, -12.3514490, -15.6416349, -12.3194189, -2.3931589, 2.3839474
7: -5.5424709, -2.0667968, -5.5636120, -2.0540648, -3.2550621, 3.2692585
8: -1.9246545, 0.3551998, -1.9464064, 0.3812060, -2.0848751, 2.0821633
9: -7.2974319, -4.0189524, -7.3074970, -4.0075884, -2.7526703, 2.7501488

Time for backsubstitution: 14.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 4666

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2769360, upper bound: 1.2801807
time: 6.96 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2769360, upper bound: 1.2826756
time: 6.86 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.0258579, -5.5622654, -9.0258694, -5.5622602, -2.8388004, 2.8530536
1: -6.5765252, -3.9591157, -6.5765290, -3.9591126, -2.2877040, 2.2805872
2: 8.3243313, 10.9319868, 8.3243275, 10.9319963, -2.3276362, 2.3231010
3: -6.1232595, -2.8826456, -6.1232624, -2.8826334, -3.0299797, 3.0246067
4: -11.8333359, -7.9824533, -11.8333511, -7.9824505, -3.0789762, 3.0832336
5: -13.6636286, -10.1825724, -13.6636314, -10.1825638, -2.6353846, 2.6278462
6: -15.6556225, -12.3172045, -15.6556263, -12.3172026, -2.4119256, 2.4194844
7: -5.5685935, -2.0476909, -5.5685978, -2.0476880, -3.3068895, 3.3007994
8: -1.9611592, 0.3840823, -1.9611712, 0.3840833, -2.1145992, 2.1270380
9: -7.3108959, -4.0054493, -7.3108988, -4.0054479, -2.7743597, 2.7761269

Time for backsubstitution: 15.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2748965, upper bound: 1.2848915
time: 15.59 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2769392, upper bound: 1.2848911
time: 9.90 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 40.76 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 40.76
Output dim: 2, lower bound: -1.2668943, upper bound: 1.2714600
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 40.76
Output dim: 2, lower bound: -1.2689667, upper bound: 1.2714597
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 40.76
Output dim: 2, lower bound: -1.2693869, upper bound: 1.2714599
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 40.76
Output dim: 2, lower bound: -1.2714598, upper bound: 1.2714619
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 40.76
Output dim: 2, lower bound: -1.2763792, upper bound: 1.2714567
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 40.76
Output dim: 2, lower bound: -1.2784060, upper bound: 1.2714582
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 40.76
Output dim: 2, lower bound: -1.2788721, upper bound: 1.2714560
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 40.76
Output dim: 2, lower bound: -1.2808973, upper bound: 1.2714582
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 40.76
Output dim: 2, lower bound: -1.2714562, upper bound: 1.2763791
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 40.76
Output dim: 2, lower bound: -1.2714563, upper bound: 1.2788718
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 40.76
Output dim: 2, lower bound: -1.2714562, upper bound: 1.2784062
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 40.76
Output dim: 2, lower bound: -1.2714563, upper bound: 1.2808972
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 40.76
Output dim: 2, lower bound: -1.2769360, upper bound: 1.2801807
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 40.76
Output dim: 2, lower bound: -1.2769360, upper bound: 1.2826756
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 40.76
Output dim: 2, lower bound: -1.2748965, upper bound: 1.2848915
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 40.76
Output dim: 2, lower bound: -1.2769392, upper bound: 1.2848911

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.9886036, -5.5828686, -8.9827385, -5.5812607, -2.7609997, 2.7533469
1: -6.5437040, -4.0016475, -6.5572376, -3.9906237, -2.2007418, 2.2050288
2: 8.3848429, 10.8714924, 8.3768234, 10.8756733, -2.2292476, 2.2316365
3: -6.0871124, -2.9179335, -6.0911970, -2.9170516, -2.9350300, 2.9383025
4: -11.8044758, -8.0108137, -11.8070097, -8.0088024, -3.0339947, 3.0346565
5: -13.6309938, -10.1993341, -13.6322517, -10.2008057, -2.5512590, 2.5538173
6: -15.6047459, -12.3426237, -15.6022434, -12.3357887, -2.3533764, 2.3422790
7: -5.5248766, -2.0957308, -5.5365758, -2.0880899, -3.2055278, 3.2127042
8: -1.9295330, 0.3734818, -1.9332438, 0.3754277, -2.0762949, 2.0794592
9: -7.2795963, -4.0319123, -7.2807961, -4.0276251, -2.7259254, 2.7236433

Time for backsubstitution: 15.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6111

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2668933, upper bound: 1.2697864
time: 12.57 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2668933, upper bound: 1.2714589
time: 7.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.9955311, -5.5823236, -8.9999638, -5.5627737, -2.7876472, 2.7639966
1: -6.5446210, -3.9989901, -6.5667076, -3.9820898, -2.2093978, 2.2187779
2: 8.3841276, 10.8751888, 8.3614788, 10.8851662, -2.2378550, 2.2512379
3: -6.0884385, -2.9145417, -6.1066170, -2.9071660, -2.9424500, 2.9570456
4: -11.8051138, -8.0099001, -11.8097439, -8.0013266, -3.0424333, 3.0385351
5: -13.6317501, -10.1964664, -13.6454201, -10.1944036, -2.5571728, 2.5703807
6: -15.6130676, -12.3421001, -15.6224937, -12.3125486, -2.3768036, 2.3546710
7: -5.5273447, -2.0914838, -5.5503931, -2.0771508, -3.2165785, 3.2316194
8: -1.9350648, 0.3739781, -1.9478607, 0.3916588, -2.0992765, 2.0908599
9: -7.2819700, -4.0306053, -7.2877431, -4.0187559, -2.7376499, 2.7309408

Time for backsubstitution: 14.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6111

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2689659, upper bound: 1.2697860
time: 12.30 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2689658, upper bound: 1.2714589
time: 8.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -8.9916773, -5.5804405, -8.9841137, -5.5810385, -2.7641516, 2.7565961
1: -6.5588512, -3.9766421, -6.5578442, -3.9795079, -2.2286386, 2.2051749
2: 8.3702278, 10.8801069, 8.3710060, 10.8760738, -2.2338562, 2.2479315
3: -6.0975232, -2.9133265, -6.0960627, -2.9169996, -2.9371257, 2.9477711
4: -11.8081341, -8.0052633, -11.8074512, -8.0062675, -3.0402899, 3.0374556
5: -13.6344681, -10.1975021, -13.6336327, -10.2006359, -2.5547352, 2.5575061
6: -15.6174240, -12.3348074, -15.6083431, -12.3353758, -2.3537655, 2.3571048
7: -5.5399523, -2.0722713, -5.5372553, -2.0768859, -3.2338781, 3.2137098
8: -1.9406214, 0.3784432, -1.9345741, 0.3778982, -2.0912781, 2.0802422
9: -7.2867374, -4.0255651, -7.2841558, -4.0269938, -2.7284460, 2.7337360

Time for backsubstitution: 15.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6111

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2693859, upper bound: 1.2697862
time: 15.96 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.2693859, upper bound: 1.2714591
time: 12.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -8.9986010, -5.5798941, -9.0013304, -5.5625515, -2.7908096, 2.7672567
1: -6.5597706, -3.9739945, -6.5673180, -3.9709826, -2.2373004, 2.2189281
2: 8.3695068, 10.8838072, 8.3556728, 10.8855600, -2.2424579, 2.2675390
3: -6.0988483, -2.9099345, -6.1115184, -2.9071150, -2.9445467, 2.9665203
4: -11.8087711, -8.0043516, -11.8101845, -7.9987774, -3.0487347, 3.0413356
5: -13.6352205, -10.1946402, -13.6468134, -10.1942329, -2.5606489, 2.5740776
6: -15.6257353, -12.3342838, -15.6285839, -12.3121376, -2.3762615, 2.3694787
7: -5.5424199, -2.0680327, -5.5510726, -2.0659630, -3.2449131, 3.2326312
8: -1.9461541, 0.3789392, -1.9491925, 0.3941293, -2.1109111, 2.0916462
9: -7.2891121, -4.0242591, -7.2911005, -4.0181236, -2.7401695, 2.7410431

Time for backsubstitution: 14.81 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=2.3431549072265625
rel_dist={2: [-1.2849368558531555, 1.2849364906580067]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4639
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1648064, upper bound: 1.1577658
time: 9.15 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688669, upper bound: 1.1688656
time: 5.86 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 15.25 seconds
IS_A1, status: Status.VERIFIED, split count: 1, time: 15.25
Output dim: 2, lower bound: -1.1648064, upper bound: 1.1577658
IS_A2, status: Status.UNKNOWN, split count: 1, time: 15.25
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

Time for backsubstitution: 15.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4639

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1577636, upper bound: 1.1648058
time: 6.73 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1577635, upper bound: 1.1688668
time: 7.36 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 29.46 seconds
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 29.46
Output dim: 2, lower bound: -1.1577636, upper bound: 1.1648058
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 29.46
Output dim: 2, lower bound: -1.1577635, upper bound: 1.1688668

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -9.0258865, -5.5622592, -9.0258865, -5.5622592, -2.7596397, 2.7596397
1: -6.5765324, -3.9591095, -6.5765324, -3.9591095, -2.2108388, 2.2108388
2: 8.3243237, 10.9320021, 8.3243237, 10.9320021, -2.2531266, 2.2531266
3: -6.1232662, -2.8826249, -6.1232662, -2.8826249, -2.9423184, 2.9423184
4: -11.8333654, -7.9824467, -11.8333654, -7.9824467, -2.9800296, 2.9800291
5: -13.6636353, -10.1825542, -13.6636353, -10.1825542, -2.5126886, 2.5126891
6: -15.6556358, -12.3172054, -15.6556358, -12.3172054, -2.3277559, 2.3277564
7: -5.5686011, -2.0476840, -5.5686011, -2.0476840, -3.2462764, 3.2462769
8: -1.9611835, 0.3840857, -1.9611835, 0.3840857, -2.0765047, 2.0765047
9: -7.3109016, -4.0054460, -7.3109016, -4.0054460, -2.7002153, 2.7002153

Time for backsubstitution: 14.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1577610, upper bound: 1.1672398
time: 13.48 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1577610, upper bound: 1.1688626
time: 17.14 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 45.75 seconds
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 45.75
Output dim: 2, lower bound: -1.1577610, upper bound: 1.1672398
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 45.75
Output dim: 2, lower bound: -1.1577610, upper bound: 1.1688626

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.9921589, -5.5871921, -9.0120773, -5.5636330, -2.7180910, 2.7120070
1: -6.5579267, -3.9706440, -6.5722828, -3.9639819, -2.1705246, 2.1837847
2: 8.3504820, 10.9175768, 8.3284235, 10.9248571, -2.2089348, 2.2279096
3: -6.0933533, -2.9072607, -6.1180687, -2.8912544, -2.9010687, 2.9161463
4: -11.8089209, -8.0150442, -11.8200426, -7.9874840, -2.9503756, 2.9332786
5: -13.6460323, -10.1961775, -13.6604385, -10.1901169, -2.4797916, 2.4926438
6: -15.6190720, -12.3514490, -15.6389666, -12.3198586, -2.2843013, 2.2717247
7: -5.5424709, -2.0667968, -5.5626535, -2.0552762, -3.1978526, 3.2116704
8: -1.9246545, 0.3551998, -1.9435835, 0.3806415, -2.0261302, 2.0207686
9: -7.2974319, -4.0189524, -7.3068528, -4.0079932, -2.6776738, 2.6749773

Time for backsubstitution: 15.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4666

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1629085, upper bound: 1.1653668
time: 10.11 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1629085, upper bound: 1.1672347
time: 13.04 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.0258579, -5.5622654, -9.0258675, -5.5622616, -2.7446518, 2.7596149
1: -6.5765252, -3.9591157, -6.5765285, -3.9591126, -2.2158904, 2.2087243
2: 8.3243313, 10.9319868, 8.3243284, 10.9319935, -2.2481909, 2.2426977
3: -6.1232595, -2.8826456, -6.1232615, -2.8826365, -2.9461198, 2.9407125
4: -11.8333359, -7.9824533, -11.8333473, -7.9824495, -2.9699950, 2.9729140
5: -13.6636286, -10.1825724, -13.6636295, -10.1825628, -2.5180397, 2.5104513
6: -15.6556225, -12.3172045, -15.6556273, -12.3172045, -2.3007154, 2.3101444
7: -5.5685935, -2.0476909, -5.5685978, -2.0476897, -3.2505903, 3.2444606
8: -1.9611592, 0.3840823, -1.9611697, 0.3840828, -2.0548768, 2.0675170
9: -7.3108959, -4.0054493, -7.3108997, -4.0054469, -2.6996832, 2.7014627

Time for backsubstitution: 15.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1613895, upper bound: 1.1688611
time: 6.22 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1629111, upper bound: 1.1688598
time: 16.53 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 38.04 seconds
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 38.04
Output dim: 2, lower bound: -1.1629085, upper bound: 1.1653668
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 38.04
Output dim: 2, lower bound: -1.1629085, upper bound: 1.1672347
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 38.04
Output dim: 2, lower bound: -1.1613895, upper bound: 1.1688611
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 38.04
Output dim: 2, lower bound: -1.1629111, upper bound: 1.1688598

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.9921579, -5.5871916, -9.0120726, -5.5636325, -2.7172813, 2.7116518
1: -6.5579271, -3.9706559, -6.5722790, -3.9640033, -2.1431289, 2.1837640
2: 8.3504896, 10.9175758, 8.3284359, 10.9248571, -2.2072401, 2.2161300
3: -6.0933456, -2.9072614, -6.1180573, -2.8912544, -2.9010611, 2.9074392
4: -11.8089180, -8.0150480, -11.8200407, -7.9874926, -2.9469662, 2.9332738
5: -13.6460323, -10.1961775, -13.6604338, -10.1901169, -2.4797888, 2.4924273
6: -15.6190662, -12.3514500, -15.6389523, -12.3198614, -2.2797937, 2.2570481
7: -5.5424700, -2.0668151, -5.5626507, -2.0553088, -3.1730213, 3.2102671
8: -1.9246531, 0.3551970, -1.9435811, 0.3806376, -2.0204582, 2.0207629
9: -7.2974281, -4.0189519, -7.3068466, -4.0079956, -2.6776686, 2.6687655

Time for backsubstitution: 14.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1629058, upper bound: 1.1657099
time: 7.71 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1629058, upper bound: 1.1672304
time: 8.12 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.0173798, -5.5629420, -9.0113697, -5.5634236, -2.7345924, 2.7440891
1: -6.5753961, -3.9623604, -6.5745955, -3.9646461, -2.2082906, 2.2029514
2: 8.3252068, 10.9274483, 8.3258228, 10.9242439, -2.2388611, 2.2358799
3: -6.1216712, -2.8867974, -6.1205330, -2.8897033, -2.9376345, 2.9341459
4: -11.8325443, -7.9835744, -11.8320131, -7.9843755, -2.9671297, 2.9702914
5: -13.6626892, -10.1860857, -13.6620216, -10.1885691, -2.5101318, 2.5045824
6: -15.6454334, -12.3178520, -15.6382065, -12.3183126, -2.2865856, 2.2911160
7: -5.5656185, -2.0528882, -5.5635109, -2.0565717, -3.2385058, 3.2336488
8: -1.9543777, 0.3834758, -1.9495754, 0.3830433, -2.0468903, 2.0547757
9: -7.3079834, -4.0070128, -7.3059273, -4.0081239, -2.6927772, 2.6940742

Time for backsubstitution: 14.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4666

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1595172, upper bound: 1.1688557
time: 11.14 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1613842, upper bound: 1.1688557
time: 6.17 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -9.0258436, -5.5622654, -9.0286274, -5.5436325, -2.7644920, 2.7541227
1: -6.5765243, -3.9591227, -6.5844111, -3.9561028, -2.2169552, 2.2178268
2: 8.3243332, 10.9319801, 8.3098307, 10.9338760, -2.2473383, 2.2522182
3: -6.1232567, -2.8826501, -6.1366549, -2.8799553, -2.9447594, 2.9541874
4: -11.8333330, -7.9824533, -11.8350763, -7.9767632, -2.9740114, 2.9748101
5: -13.6636295, -10.1825743, -13.6758003, -10.1821461, -2.5158443, 2.5202851
6: -15.6555986, -12.3172054, -15.6584587, -12.2935295, -2.3031201, 2.3024333
7: -5.5685883, -2.0476999, -5.5774841, -2.0455108, -3.2498779, 3.2537088
8: -1.9611487, 0.3840818, -1.9643140, 0.3993130, -2.0618341, 2.0658607
9: -7.3108902, -4.0054517, -7.3130589, -3.9991078, -2.7052121, 2.7016330

Time for backsubstitution: 14.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4666

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1610383, upper bound: 1.1688556
time: 7.59 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1629057, upper bound: 1.1688556
time: 18.73 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 41.53 seconds
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 41.53
Output dim: 2, lower bound: -1.1629058, upper bound: 1.1657099
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 41.53
Output dim: 2, lower bound: -1.1629058, upper bound: 1.1672304
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 41.53
Output dim: 2, lower bound: -1.1595172, upper bound: 1.1688557
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 41.53
Output dim: 2, lower bound: -1.1613842, upper bound: 1.1688557
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 41.53
Output dim: 2, lower bound: -1.1610383, upper bound: 1.1688556
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 41.53
Output dim: 2, lower bound: -1.1629057, upper bound: 1.1688556

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -8.9949369, -5.5698690, -9.0120573, -5.5636344, -2.7117786, 2.7297430
1: -6.5654125, -3.9676490, -6.5722790, -3.9640107, -2.1516991, 2.1848145
2: 8.3366489, 10.9193411, 8.3284359, 10.9248476, -2.2120466, 2.2150016
3: -6.1058817, -2.9046016, -6.1180544, -2.8912597, -2.9138427, 2.9061155
4: -11.8103094, -8.0095062, -11.8200417, -7.9874926, -2.9484749, 2.9375248
5: -13.6575336, -10.1957836, -13.6604328, -10.1901188, -2.4911604, 2.4902034
6: -15.6220074, -12.3292961, -15.6389294, -12.3198633, -2.2721243, 2.2578409
7: -5.5511227, -2.0646284, -5.5626459, -2.0553160, -3.1820908, 3.2095470
8: -1.9277668, 0.3704205, -1.9435711, 0.3806362, -2.0186462, 2.0238891
9: -7.2994714, -4.0128260, -7.3068409, -4.0079985, -2.6778107, 2.6740847

Time for backsubstitution: 14.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6111

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1615962, upper bound: 1.1672315
time: 15.54 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1629051, upper bound: 1.1672300
time: 10.67 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -9.0142136, -5.5671988, -9.0096960, -5.5647764, -2.7296505, 2.7374039
1: -6.5598178, -3.9874163, -6.5736175, -3.9778023, -2.1773500, 2.1764641
2: 8.3407154, 10.9186211, 8.3332262, 10.9236412, -2.2223649, 2.2184324
3: -6.1102934, -2.8914125, -6.1142035, -2.8897679, -2.9265594, 2.9233890
4: -11.8284044, -7.9892964, -11.8312025, -7.9874763, -2.9593773, 2.9637263
5: -13.6583519, -10.1879444, -13.6598682, -10.1887875, -2.5054722, 2.5000749
6: -15.6326885, -12.3277941, -15.6309652, -12.3200788, -2.2718561, 2.2709525
7: -5.5501647, -2.0765824, -5.5624685, -2.0699482, -3.2057891, 3.2088890
8: -1.9430747, 0.3784509, -1.9478731, 0.3800869, -2.0315170, 2.0482185
9: -7.3005738, -4.0136609, -7.3017979, -4.0090466, -2.6837664, 2.6828594

Time for backsubstitution: 14.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6111

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1582081, upper bound: 1.1688546
time: 14.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1595165, upper bound: 1.1688548
time: 16.25 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -9.0173712, -5.5629444, -9.0113688, -5.5634270, -2.7342815, 2.7432966
1: -6.5753942, -3.9623837, -6.5745955, -3.9646585, -2.2082644, 2.1756499
2: 8.3252144, 10.9274473, 8.3258305, 10.9242430, -2.2251711, 2.2316852
3: -6.1216583, -2.8867970, -6.1205273, -2.8897042, -2.9289160, 2.9341388
4: -11.8325434, -7.9835811, -11.8320112, -7.9843807, -2.9651523, 2.9662857
5: -13.6626873, -10.1860828, -13.6620197, -10.1885729, -2.5099134, 2.5045815
6: -15.6454191, -12.3178568, -15.6382008, -12.3183117, -2.2720122, 2.2839115
7: -5.5656176, -2.0529203, -5.5635099, -2.0565891, -3.2371130, 3.2088547
8: -1.9543743, 0.3834720, -1.9495730, 0.3830404, -2.0449929, 2.0470424
9: -7.3079758, -4.0070148, -7.3059244, -4.0081239, -2.6865811, 2.6940689

Time for backsubstitution: 14.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6111

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1600758, upper bound: 1.1688547
time: 8.35 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1613834, upper bound: 1.1688572
time: 6.78 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -9.0226879, -5.5665207, -9.0269623, -5.5449848, -2.7595615, 2.7474275
1: -6.5609460, -3.9841664, -6.5834274, -3.9692535, -2.1860094, 2.1913517
2: 8.3398476, 10.9231482, 8.3172207, 10.9332867, -2.2308464, 2.2347372
3: -6.1118803, -2.8872654, -6.1303072, -2.8800216, -2.9336820, 2.9434152
4: -11.8291912, -7.9881797, -11.8342705, -7.9798813, -2.9659219, 2.9682474
5: -13.6592884, -10.1844349, -13.6736393, -10.1823616, -2.5111876, 2.5154302
6: -15.6428709, -12.3271503, -15.6512299, -12.2952986, -2.2884145, 2.2822728
7: -5.5531340, -2.0713840, -5.5764432, -2.0588689, -3.2171822, 3.2289557
8: -1.9498444, 0.3790565, -1.9626102, 0.3963547, -2.0464637, 2.0593426
9: -7.3034816, -4.0121002, -7.3089337, -4.0000315, -2.6962018, 2.6904092

Time for backsubstitution: 15.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6111

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1597287, upper bound: 1.1688551
time: 9.67 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1610375, upper bound: 1.1688550
time: 8.35 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -9.0258389, -5.5622683, -9.0286274, -5.5436354, -2.7642097, 2.7533274
1: -6.5765214, -3.9591441, -6.5844097, -3.9561157, -2.2169299, 2.1905437
2: 8.3243446, 10.9319801, 8.3098354, 10.9338760, -2.2337089, 2.2445810
3: -6.1232433, -2.8826499, -6.1366482, -2.8799558, -2.9360409, 2.9541788
4: -11.8333321, -7.9824605, -11.8350782, -7.9767656, -2.9716902, 2.9708059
5: -13.6636238, -10.1825733, -13.6758003, -10.1821461, -2.5156288, 2.5186460
6: -15.6555834, -12.3172073, -15.6584530, -12.2935314, -2.2885745, 2.2952294
7: -5.5685863, -2.0477321, -5.5774832, -2.0455289, -3.2484846, 3.2289324
8: -1.9611454, 0.3840775, -1.9643126, 0.3993096, -2.0568817, 2.0581195
9: -7.3108854, -4.0054541, -7.3130574, -3.9991078, -2.6990175, 2.7016277

Time for backsubstitution: 15.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6111

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1615961, upper bound: 1.1688548
time: 30.15 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1629050, upper bound: 1.1688544
time: 10.44 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 55.90 seconds
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 55.90
Output dim: 2, lower bound: -1.1615962, upper bound: 1.1672315
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 55.90
Output dim: 2, lower bound: -1.1629051, upper bound: 1.1672300
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 55.90
Output dim: 2, lower bound: -1.1582081, upper bound: 1.1688546
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 55.90
Output dim: 2, lower bound: -1.1595165, upper bound: 1.1688548
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 55.90
Output dim: 2, lower bound: -1.1600758, upper bound: 1.1688547
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 55.90
Output dim: 2, lower bound: -1.1613834, upper bound: 1.1688572
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 55.90
Output dim: 2, lower bound: -1.1597287, upper bound: 1.1688551
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 55.90
Output dim: 2, lower bound: -1.1610375, upper bound: 1.1688550
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 55.90
Output dim: 2, lower bound: -1.1615961, upper bound: 1.1688548
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 55.90
Output dim: 2, lower bound: -1.1629050, upper bound: 1.1688544

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -8.9925184, -5.5787754, -8.9928493, -5.5793610, -2.6934719, 2.7012815
1: -6.5607142, -3.9690599, -6.5583067, -3.9709573, -2.1390057, 2.1678853
2: 8.3384323, 10.9140987, 8.3431673, 10.9148006, -2.1964974, 2.1915233
3: -6.1005363, -2.9145482, -6.0936909, -2.9106541, -2.8894358, 2.8724456
4: -11.8048439, -8.0129576, -11.8100710, -8.0005980, -2.9255075, 2.9243846
5: -13.6538086, -10.1963596, -13.6512775, -10.1932058, -2.4846325, 2.4801435
6: -15.6209774, -12.3388271, -15.6253319, -12.3369265, -2.2525742, 2.2290196
7: -5.5290213, -2.0656915, -5.5208611, -2.0724502, -3.1425238, 3.1667385
8: -1.9259925, 0.3689909, -1.9362001, 0.3756628, -2.0080333, 2.0096407
9: -7.2962642, -4.0165715, -7.2991681, -4.0177603, -2.6640306, 2.6620531

Time for backsubstitution: 14.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6191

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1615961, upper bound: 1.1668115
time: 15.14 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1615961, upper bound: 1.1672335
time: 10.65 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -8.9949322, -5.5698729, -9.0120544, -5.5636415, -2.7085075, 2.7244825
1: -6.5654087, -3.9676495, -6.5722713, -3.9640119, -2.1516633, 2.1814394
2: 8.3366499, 10.9193382, 8.3284397, 10.9248409, -2.2028203, 2.2025890
3: -6.1058764, -2.9046090, -6.1180487, -2.8912697, -2.8949537, 2.9060993
4: -11.8103056, -8.0095081, -11.8200350, -7.9874959, -2.9433460, 2.9317443
5: -13.6575336, -10.1957846, -13.6604280, -10.1901178, -2.4887519, 2.4893165
6: -15.6220074, -12.3292990, -15.6389294, -12.3198719, -2.2583687, 2.2489269
7: -5.5511084, -2.0646284, -5.5626230, -2.0553179, -3.1820683, 3.1984653
8: -1.9277654, 0.3704190, -1.9435682, 0.3806343, -2.0112972, 2.0208149
9: -7.2994690, -4.0128293, -7.3068352, -4.0080018, -2.6778007, 2.6743631

Time for backsubstitution: 14.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 4666
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6191

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1629049, upper bound: 1.1668090
time: 8.98 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1629049, upper bound: 1.1672298
time: 7.07 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -9.0117731, -5.5761065, -8.9904737, -5.5804982, -2.7113266, 2.7089200
1: -6.5551243, -3.9888120, -6.5596685, -3.9847808, -2.1647196, 2.1595190
2: 8.3425159, 10.9133492, 8.3480167, 10.9135647, -2.2067471, 2.1950717
3: -6.1049323, -2.9013429, -6.0898223, -2.9091575, -2.9021988, 2.8897238
4: -11.8229427, -7.9927387, -11.8212404, -8.0005808, -2.9364080, 2.9506021
5: -13.6546316, -10.1885214, -13.6507025, -10.1918783, -2.4989462, 2.4900126
6: -15.6316338, -12.3373299, -15.6173687, -12.3371372, -2.2522933, 2.2421451
7: -5.5280428, -2.0776532, -5.5207090, -2.0870953, -3.1662655, 3.1661038
8: -1.9412837, 0.3770165, -1.9405231, 0.3751063, -2.0208864, 2.0340352
9: -7.2973394, -4.0174098, -7.2941055, -4.0187912, -2.6699748, 2.6708069

Time for backsubstitution: 14.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6191

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1582080, upper bound: 1.1684315
time: 6.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1582080, upper bound: 1.1688547
time: 8.22 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -9.0142126, -5.5672030, -9.0096922, -5.5647860, -2.7263794, 2.7373915
1: -6.5598149, -3.9874177, -6.5736113, -3.9778047, -2.1773300, 2.1731124
2: 8.3407183, 10.9186192, 8.3332291, 10.9236364, -2.2135696, 2.2054431
3: -6.1102896, -2.8914185, -6.1141949, -2.8897758, -2.9076705, 2.9233742
4: -11.8283997, -7.9892974, -11.8311968, -7.9874802, -2.9515543, 2.9579728
5: -13.6583500, -10.1879463, -13.6598682, -10.1887903, -2.5035310, 2.4991837
6: -15.6326857, -12.3277998, -15.6309652, -12.3200893, -2.2580609, 2.2620397
7: -5.5501499, -2.0765820, -5.5624461, -2.0699508, -3.2057676, 3.1978068
8: -1.9430737, 0.3784509, -1.9478712, 0.3800840, -2.0241737, 2.0451343
9: -7.3005724, -4.0136652, -7.3017950, -4.0090528, -2.6837568, 2.6831365

Time for backsubstitution: 14.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6191

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1595163, upper bound: 1.1684313
time: 6.04 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1595163, upper bound: 1.1688548
time: 8.24 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -9.0149250, -5.5718536, -8.9921436, -5.5791464, -2.7159557, 2.7148061
1: -6.5706973, -3.9637928, -6.5606337, -3.9716153, -2.1956248, 2.1587081
2: 8.3270350, 10.9221678, 8.3405933, 10.9141655, -2.2095985, 2.2084453
3: -6.1162920, -2.8967259, -6.0961494, -2.9090936, -2.9045515, 2.9004750
4: -11.8270864, -7.9870186, -11.8220463, -7.9974842, -2.9421725, 2.9531560
5: -13.6589718, -10.1866589, -13.6528540, -10.1916599, -2.5033922, 2.4945240
6: -15.6443653, -12.3273907, -15.6246042, -12.3353729, -2.2524405, 2.2551048
7: -5.5435009, -2.0539925, -5.5217462, -2.0737269, -3.1975865, 3.1660600
8: -1.9525933, 0.3820367, -1.9422026, 0.3780613, -2.0343089, 2.0328693
9: -7.3047447, -4.0107555, -7.2982335, -4.0178766, -2.6727891, 2.6820297

Time for backsubstitution: 14.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6191

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1600755, upper bound: 1.1684334
time: 7.86 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1600755, upper bound: 1.1688548
time: 10.13 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -9.0173731, -5.5629482, -9.0113640, -5.5634360, -2.7310123, 2.7429214
1: -6.5753899, -3.9623847, -6.5745878, -3.9646602, -2.2082367, 2.1722789
2: 8.3252172, 10.9274454, 8.3258324, 10.9242373, -2.2158909, 2.2153342
3: -6.1216531, -2.8868015, -6.1205196, -2.8897123, -2.9100270, 2.9341235
4: -11.8325405, -7.9835806, -11.8320084, -7.9843845, -2.9573283, 2.9605281
5: -13.6626835, -10.1860857, -13.6620131, -10.1885710, -2.5069556, 2.5036936
6: -15.6454191, -12.3178625, -15.6381989, -12.3183231, -2.2582154, 2.2749979
7: -5.5656033, -2.0529201, -5.5634875, -2.0565906, -3.2363472, 3.1977735
8: -1.9543729, 0.3834715, -1.9495711, 0.3830404, -2.0366952, 2.0439680
9: -7.3079739, -4.0070171, -7.3059192, -4.0081296, -2.6865716, 2.6943469

Time for backsubstitution: 14.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6191

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1613832, upper bound: 1.1684312
time: 5.94 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1613833, upper bound: 1.1688551
time: 11.62 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -9.0202465, -5.5754313, -9.0077105, -5.5607023, -2.7412395, 2.7189312
1: -6.5562515, -3.9855623, -6.5694857, -3.9762387, -2.1733856, 2.1744418
2: 8.3416481, 10.9178782, 8.3320913, 10.9232073, -2.2152214, 2.2114406
3: -6.1065121, -2.8971975, -6.1059327, -2.8993847, -2.9093962, 2.9097490
4: -11.8237286, -7.9916143, -11.8243122, -7.9929996, -2.9429502, 2.9551306
5: -13.6555710, -10.1850109, -13.6644869, -10.1854515, -2.5046616, 2.5053093
6: -15.6418142, -12.3366871, -15.6376019, -12.3123550, -2.2688460, 2.2534778
7: -5.5310173, -2.0724535, -5.5346932, -2.0760183, -3.1776304, 3.1861830
8: -1.9480543, 0.3776207, -1.9553270, 0.3913741, -2.0357854, 2.0450513
9: -7.3002448, -4.0158467, -7.3012362, -4.0097642, -2.6824279, 2.6783643

Time for backsubstitution: 14.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6191

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1597285, upper bound: 1.1645184
time: 11.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1597285, upper bound: 1.1688546
time: 9.55 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -9.0226870, -5.5665250, -9.0269585, -5.5449953, -2.7537446, 2.7474170
1: -6.5609417, -3.9841661, -6.5834208, -3.9692557, -2.1859989, 2.1880057
2: 8.3398476, 10.9231472, 8.3172207, 10.9332829, -2.2221022, 2.2183859
3: -6.1118755, -2.8872719, -6.1302996, -2.8800309, -2.9147925, 2.9395847
4: -11.8291903, -7.9881830, -11.8342638, -7.9798818, -2.9580965, 2.9624970
5: -13.6592884, -10.1844358, -13.6736355, -10.1823635, -2.5092626, 2.5128219
6: -15.6428699, -12.3271580, -15.6512299, -12.2953043, -2.2746174, 2.2733607
7: -5.5531187, -2.0713844, -5.5764189, -2.0588717, -3.2171602, 3.2178750
8: -1.9498439, 0.3790545, -1.9626079, 0.3963532, -2.0381773, 2.0561993
9: -7.3034792, -4.0121050, -7.3089285, -4.0000372, -2.6961937, 2.6906867

Time for backsubstitution: 14.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 6191
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 4616
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 520
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6191

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1610373, upper bound: 1.1684312
time: 5.88 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1610374, upper bound: 1.1688570
time: 7.40 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -9.0233936, -5.5711727, -9.0093708, -5.5593529, -2.7458906, 2.7248259
1: -6.5718222, -3.9605546, -6.5704565, -3.9630804, -2.2042952, 2.1736414
2: 8.3261623, 10.9266968, 8.3246803, 10.9237976, -2.2181292, 2.2213321
3: -6.1178770, -2.8925784, -6.1122775, -2.8993213, -2.9117498, 2.9205174
4: -11.8278770, -7.9858952, -11.8251171, -7.9898834, -2.9487081, 2.9576826
5: -13.6599064, -10.1831493, -13.6666527, -10.1852322, -2.5091066, 2.5085351
6: -15.6545296, -12.3267422, -15.6448259, -12.3105898, -2.2689993, 2.2664323
7: -5.5464754, -2.0488045, -5.5357318, -2.0626719, -3.2089367, 3.1861467
8: -1.9593625, 0.3826408, -1.9570122, 0.3943319, -2.0461950, 2.0438781
9: -7.3076491, -4.0091934, -7.3053627, -4.0088496, -2.6852407, 2.6895957

Time for backsubstitution: 14.50 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=2.2642369270324707
rel_dist={2: [-1.168894797061638, 1.1688945587998152]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2422.03 seconds
