## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_5.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 0.045175422


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553)
1: (-0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822)
2: (-0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808)
3: (-0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506)
4: (-0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249)

## BASE Result
execution time: IAR + LP analysis = 1.82 + 0.85 = 2.68 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0465726, upper bound: 0.0465726


# Binary Search by BASE starts (time budget: 1197.32 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.0909091


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0909091, mid=0.0909091, abs_max=0.05155529826879501
rel_dist={0: [-0.04657254158633466, 0.04657254158633465]}

## Binary search (step 1) starts
Candidate diff: 0.0454545


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0454545, mid=0.0454545, abs_max=0.05155529826879501
rel_dist={0: [-0.04655466850732895, 0.046554668507328936]}

## Binary search (step 2) starts
Candidate diff: 0.0227273


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0227273, mid=0.0227273, abs_max=0.05155529826879501
rel_dist={0: [-0.04643723777492976, 0.04643723777492974]}

## Binary search (step 3) starts
Candidate diff: 0.0113636


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0113636, mid=0.0113636, abs_max=0.05155529826879501
rel_dist={0: [-0.046257134670243685, 0.046257134670243644]}

## Binary search (step 4) starts
Candidate diff: 0.0056818


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0056818, mid=0.0056818, abs_max=0.05155529826879501
rel_dist={0: [-0.046105128879509696, 0.04610512888044604]}

## Binary search (step 5) starts
Candidate diff: 0.0028409


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0028409, mid=0.0028409, abs_max=0.05155529826879501
rel_dist={0: [-0.045911576120675604, 0.04591157612368746]}

## Binary search (step 6) starts
Candidate diff: 0.0014205


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0014205, mid=0.0014205, abs_max=0.05155529826879501
rel_dist={0: [-0.04568732200225989, 0.045687322002010186]}

## Binary search (step 7) starts
Candidate diff: 0.0007102


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0007102, mid=0.0007102, abs_max=0.05155529826879501
rel_dist={0: [-0.045566110620896404, 0.045566110621249975]}

## Binary search (step 8) starts
Candidate diff: 0.0003551


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0003551, mid=0.0003551, abs_max=0.05155529826879501
rel_dist={0: [-0.045504536918878964, 0.045504536919091856]}

## Binary search (step 9) starts
Candidate diff: 0.0001776


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0001776, mid=0.0001776, abs_max=0.05155529826879501
rel_dist={0: [-0.04547234088213036, 0.045472340882226694]}

## Binary search (step 10) starts
Candidate diff: 0.0000888


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0000888, mid=0.0000888, abs_max=0.05155529826879501
rel_dist={0: [-0.045455803727991226, 0.04545580372804092]}

## Binary search (step 11) starts
Candidate diff: 0.0000444


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000444, mid=0.0000444, abs_max=0.05155529826879501
rel_dist={0: [-0.045447531203799935, 0.04544753229716292]}

## Binary search (step 12) starts
Candidate diff: 0.0000222


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000222, mid=0.0000222, abs_max=0.05155529826879501
rel_dist={0: [-0.04544327253165521, 0.045443272531668105]}

## Binary search (step 13) starts
Candidate diff: 0.0000111


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000111, mid=0.0000111, abs_max=0.05155529826879501
rel_dist={0: [-0.045441142812514546, 0.0454411253591801]}

## Binary search (step 14) starts
Candidate diff: 0.0000055


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000055, mid=0.0000055, abs_max=0.05155529826879501
rel_dist={0: [-0.045440078245859696, 0.045440078245863005]}

## Binary search (step 15) starts
Candidate diff: 0.0000028


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000028, mid=0.0000028, abs_max=0.05155529826879501
rel_dist={0: [-0.04543954642833681, 0.04543954642833836]}

## Binary search (step 16) starts
Candidate diff: 0.0000014


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000014, mid=0.0000014, abs_max=0.05155529826879501
rel_dist={0: [-0.04543928115040508, 0.045439281150405936]}

## Binary search (step 17) starts
Candidate diff: 0.0000007


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000007, mid=0.0000007, abs_max=0.05155529826879501
rel_dist={0: [-0.045439151719456165, 0.04543914919181363]}

## Binary Search Result
Binary search time: 45.68 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1151.65 seconds

## Binary search (step 0) starts
Candidate diff: 0.0909091


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462643, upper bound: 0.0465606
time: 0.31 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462653, upper bound: 0.0462653
time: 0.28 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.73 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.0462643, upper bound: 0.0465606
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.0462653, upper bound: 0.0462653

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0070904, 0.0444649, -0.0482198, 0.0411790
1: -0.0356151, 0.0616529, -0.0457317, 0.0752505, -0.1108656, 0.1073847
2: -0.0105594, 0.0539910, -0.0161715, 0.0679093, -0.0784687, 0.0701625
3: -0.0435620, 0.0632024, -0.0558398, 0.0794108, -0.1229728, 0.1190422
4: -0.0277857, 0.0573916, -0.0350063, 0.0803186, -0.1081043, 0.0923979

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462643, upper bound: 0.0462643
time: 0.31 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462643, upper bound: 0.0462653
time: 0.29 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0070904, 0.0444649, -0.0635343, 0.1091957
1: -0.0749285, 0.1370497, -0.0457317, 0.0752505, -0.1501790, 0.1827814
2: -0.0416235, 0.1398997, -0.0161715, 0.0679093, -0.1095328, 0.1560712
3: -0.0797155, 0.1714330, -0.0558398, 0.0794108, -0.1591263, 0.2272728
4: -0.0581345, 0.1992887, -0.0350063, 0.0803186, -0.1384531, 0.2342951

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462653, upper bound: 0.0462643
time: 0.30 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462653, upper bound: 0.0462653
time: 0.31 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.54 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 0, lower bound: -0.0462643, upper bound: 0.0462643
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 0, lower bound: -0.0462643, upper bound: 0.0462653
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 0, lower bound: -0.0462653, upper bound: 0.0462643
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 0, lower bound: -0.0462653, upper bound: 0.0462653

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0037549, 0.0340886, -0.0378435, 0.0378435
1: -0.0356151, 0.0616529, -0.0356151, 0.0616529, -0.0972680, 0.0972680
2: -0.0105594, 0.0539910, -0.0105594, 0.0539910, -0.0645503, 0.0645503
3: -0.0435620, 0.0632024, -0.0435620, 0.0632024, -0.1067644, 0.1067644
4: -0.0277857, 0.0573916, -0.0277857, 0.0573916, -0.0851773, 0.0851773

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0464070
time: 0.29 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.30 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0190694, 0.1021053, -0.1058602, 0.0531580
1: -0.0356151, 0.0616529, -0.0749285, 0.1370497, -0.1726648, 0.1365814
2: -0.0105594, 0.0539910, -0.0416235, 0.1398997, -0.1504591, 0.0956145
3: -0.0435620, 0.0632024, -0.0797155, 0.1714330, -0.2149950, 0.1429179
4: -0.0277857, 0.0573916, -0.0581345, 0.1992887, -0.2270745, 0.1155261

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0464073
time: 0.29 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.30 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0037549, 0.0340886, -0.0531580, 0.1058602
1: -0.0749285, 0.1370497, -0.0356151, 0.0616529, -0.1365814, 0.1726648
2: -0.0416235, 0.1398997, -0.0105594, 0.0539910, -0.0956145, 0.1504591
3: -0.0797155, 0.1714330, -0.0435620, 0.0632024, -0.1429179, 0.2149950
4: -0.0581345, 0.1992887, -0.0277857, 0.0573916, -0.1155261, 0.2270745

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0459994
time: 0.30 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449499, upper bound: 0.0449499
time: 0.28 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0190694, 0.1021053, -0.1211746, 0.1211746
1: -0.0749285, 0.1370497, -0.0749285, 0.1370497, -0.2119782, 0.2119782
2: -0.0416235, 0.1398997, -0.0416235, 0.1398997, -0.1815232, 0.1815232
3: -0.0797155, 0.1714330, -0.0797155, 0.1714330, -0.2511485, 0.2511485
4: -0.0581345, 0.1992887, -0.0581345, 0.1992887, -0.2574233, 0.2574233

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0459994
time: 0.30 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.28 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.90 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0464070
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.90
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0464073
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.90
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0459994
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.90
Output dim: 0, lower bound: -0.0449499, upper bound: 0.0449499
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.90
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0459994
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.90
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0037549, 0.0340886, -0.0376604, 0.0373968
1: -0.0350209, 0.0608409, -0.0356151, 0.0616529, -0.0966738, 0.0964560
2: -0.0102656, 0.0532160, -0.0105594, 0.0539910, -0.0642565, 0.0637753
3: -0.0427086, 0.0621664, -0.0435620, 0.0632024, -0.1059110, 0.1057284
4: -0.0273492, 0.0564452, -0.0277857, 0.0573916, -0.0847408, 0.0842309

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447883, upper bound: 0.0463993
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
time: 0.28 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
time: 0.27 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0190694, 0.1021053, -0.1056770, 0.0527113
1: -0.0350209, 0.0608409, -0.0749285, 0.1370497, -0.1720706, 0.1357694
2: -0.0102656, 0.0532160, -0.0416235, 0.1398997, -0.1501652, 0.0948395
3: -0.0427086, 0.0621664, -0.0797155, 0.1714330, -0.2141416, 0.1418818
4: -0.0273492, 0.0564452, -0.0581345, 0.1992887, -0.2266379, 0.1145797

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447348, upper bound: 0.0464004
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
time: 0.28 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
time: 0.30 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0037549, 0.0340886, -0.0527285, 0.1038970
1: -0.0738246, 0.1346942, -0.0356151, 0.0616529, -0.1354775, 0.1703093
2: -0.0407216, 0.1377040, -0.0105594, 0.0539910, -0.0947125, 0.1482634
3: -0.0786182, 0.1676345, -0.0435620, 0.0632024, -0.1418206, 0.2111965
4: -0.0572622, 0.1955275, -0.0277857, 0.0573916, -0.1146538, 0.2233132

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
time: 0.30 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
time: 0.29 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0190694, 0.1021053, -0.1207452, 0.1192114
1: -0.0738246, 0.1346942, -0.0749285, 0.1370497, -0.2108742, 0.2096227
2: -0.0407216, 0.1377040, -0.0416235, 0.1398997, -0.1806213, 0.1793275
3: -0.0786182, 0.1676345, -0.0797155, 0.1714330, -0.2500512, 0.2473500
4: -0.0572622, 0.1955275, -0.0581345, 0.1992887, -0.2565510, 0.2536620

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
time: 0.31 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.30 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.90 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.90
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.90
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.90
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.90
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.90
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.90
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.90
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.90
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
Binary search (step 0): status=Status.VERIFIED, low=0.0909091, high=0.1818182, mid=0.0909091, abs_max=0.05155529826879501
rel_dist={0: [-0.04657254158633466, 0.04657254158633465]}

## Binary search (step 1) starts
Candidate diff: 0.1363636


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
time: 0.29 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.31 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.78 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0070904, 0.0444649, -0.0482198, 0.0411790
1: -0.0356151, 0.0616529, -0.0457317, 0.0752505, -0.1108656, 0.1073847
2: -0.0105594, 0.0539910, -0.0161715, 0.0679093, -0.0784687, 0.0701625
3: -0.0435620, 0.0632024, -0.0558398, 0.0794108, -0.1229728, 0.1190422
4: -0.0277857, 0.0573916, -0.0350063, 0.0803186, -0.1081043, 0.0923979

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
time: 0.30 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.30 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0070904, 0.0444649, -0.0635343, 0.1091957
1: -0.0749285, 0.1370497, -0.0457317, 0.0752505, -0.1501790, 0.1827814
2: -0.0416235, 0.1398997, -0.0161715, 0.0679093, -0.1095328, 0.1560712
3: -0.0797155, 0.1714330, -0.0558398, 0.0794108, -0.1591263, 0.2272728
4: -0.0581345, 0.1992887, -0.0350063, 0.0803186, -0.1384531, 0.2342951

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
time: 0.28 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.47 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.47
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0037549, 0.0340886, -0.0378435, 0.0378435
1: -0.0356151, 0.0616529, -0.0356151, 0.0616529, -0.0972680, 0.0972680
2: -0.0105594, 0.0539910, -0.0105594, 0.0539910, -0.0645503, 0.0645503
3: -0.0435620, 0.0632024, -0.0435620, 0.0632024, -0.1067644, 0.1067644
4: -0.0277857, 0.0573916, -0.0277857, 0.0573916, -0.0851773, 0.0851773

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.27 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.27 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0190694, 0.1021053, -0.1058602, 0.0531580
1: -0.0356151, 0.0616529, -0.0749285, 0.1370497, -0.1726648, 0.1365814
2: -0.0105594, 0.0539910, -0.0416235, 0.1398997, -0.1504591, 0.0956145
3: -0.0435620, 0.0632024, -0.0797155, 0.1714330, -0.2149950, 0.1429179
4: -0.0277857, 0.0573916, -0.0581345, 0.1992887, -0.2270745, 0.1155261

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.27 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
time: 0.27 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0037549, 0.0340886, -0.0531580, 0.1058602
1: -0.0749285, 0.1370497, -0.0356151, 0.0616529, -0.1365814, 0.1726648
2: -0.0416235, 0.1398997, -0.0105594, 0.0539910, -0.0956145, 0.1504591
3: -0.0797155, 0.1714330, -0.0435620, 0.0632024, -0.1429179, 0.2149950
4: -0.0581345, 0.1992887, -0.0277857, 0.0573916, -0.1155261, 0.2270745

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0461760
time: 0.29 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.29 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0190694, 0.1021053, -0.1211746, 0.1211746
1: -0.0749285, 0.1370497, -0.0749285, 0.1370497, -0.2119782, 0.2119782
2: -0.0416235, 0.1398997, -0.0416235, 0.1398997, -0.1815232, 0.1815232
3: -0.0797155, 0.1714330, -0.0797155, 0.1714330, -0.2511485, 0.2511485
4: -0.0581345, 0.1992887, -0.0581345, 0.1992887, -0.2574233, 0.2574233

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0461760
time: 0.30 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.29 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.65 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.65
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.65
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0461760
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.65
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0461760
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.65
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0037549, 0.0340886, -0.0376604, 0.0373968
1: -0.0350209, 0.0608409, -0.0356151, 0.0616529, -0.0966738, 0.0964560
2: -0.0102656, 0.0532160, -0.0105594, 0.0539910, -0.0642565, 0.0637753
3: -0.0427086, 0.0621664, -0.0435620, 0.0632024, -0.1059110, 0.1057284
4: -0.0273492, 0.0564452, -0.0277857, 0.0573916, -0.0847408, 0.0842309

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447883, upper bound: 0.0465151
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
time: 0.29 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
time: 0.29 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0190694, 0.1021053, -0.1056770, 0.0527113
1: -0.0350209, 0.0608409, -0.0749285, 0.1370497, -0.1720706, 0.1357694
2: -0.0102656, 0.0532160, -0.0416235, 0.1398997, -0.1501652, 0.0948395
3: -0.0427086, 0.0621664, -0.0797155, 0.1714330, -0.2141416, 0.1418818
4: -0.0273492, 0.0564452, -0.0581345, 0.1992887, -0.2266379, 0.1145797

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447348, upper bound: 0.0465151
time: 0.28 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
time: 0.28 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
time: 0.27 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0037549, 0.0340886, -0.0527285, 0.1038970
1: -0.0738246, 0.1346942, -0.0356151, 0.0616529, -0.1354775, 0.1703093
2: -0.0407216, 0.1377040, -0.0105594, 0.0539910, -0.0947125, 0.1482634
3: -0.0786182, 0.1676345, -0.0435620, 0.0632024, -0.1418206, 0.2111965
4: -0.0572622, 0.1955275, -0.0277857, 0.0573916, -0.1146538, 0.2233132

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
time: 0.29 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
time: 0.27 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0190694, 0.1021053, -0.1207452, 0.1192114
1: -0.0738246, 0.1346942, -0.0749285, 0.1370497, -0.2108742, 0.2096227
2: -0.0407216, 0.1377040, -0.0416235, 0.1398997, -0.1806213, 0.1793275
3: -0.0786182, 0.1676345, -0.0797155, 0.1714330, -0.2500512, 0.2473500
4: -0.0572622, 0.1955275, -0.0581345, 0.1992887, -0.2565510, 0.2536620

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
time: 0.28 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.28 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.61 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
Binary search (step 1): status=Status.VERIFIED, low=0.1363636, high=0.1818182, mid=0.1363636, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 2) starts
Candidate diff: 0.1590909


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
time: 0.29 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.28 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.73 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0070904, 0.0444649, -0.0482198, 0.0411790
1: -0.0356151, 0.0616529, -0.0457317, 0.0752505, -0.1108656, 0.1073847
2: -0.0105594, 0.0539910, -0.0161715, 0.0679093, -0.0784687, 0.0701625
3: -0.0435620, 0.0632024, -0.0558398, 0.0794108, -0.1229728, 0.1190422
4: -0.0277857, 0.0573916, -0.0350063, 0.0803186, -0.1081043, 0.0923979

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
time: 0.29 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.32 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0070904, 0.0444649, -0.0635343, 0.1091957
1: -0.0749285, 0.1370497, -0.0457317, 0.0752505, -0.1501790, 0.1827814
2: -0.0416235, 0.1398997, -0.0161715, 0.0679093, -0.1095328, 0.1560712
3: -0.0797155, 0.1714330, -0.0558398, 0.0794108, -0.1591263, 0.2272728
4: -0.0581345, 0.1992887, -0.0350063, 0.0803186, -0.1384531, 0.2342951

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
time: 0.29 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.56 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.56
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0037549, 0.0340886, -0.0378435, 0.0378435
1: -0.0356151, 0.0616529, -0.0356151, 0.0616529, -0.0972680, 0.0972680
2: -0.0105594, 0.0539910, -0.0105594, 0.0539910, -0.0645503, 0.0645503
3: -0.0435620, 0.0632024, -0.0435620, 0.0632024, -0.1067644, 0.1067644
4: -0.0277857, 0.0573916, -0.0277857, 0.0573916, -0.0851773, 0.0851773

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.28 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.30 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0190694, 0.1021053, -0.1058602, 0.0531580
1: -0.0356151, 0.0616529, -0.0749285, 0.1370497, -0.1726648, 0.1365814
2: -0.0105594, 0.0539910, -0.0416235, 0.1398997, -0.1504591, 0.0956145
3: -0.0435620, 0.0632024, -0.0797155, 0.1714330, -0.2149950, 0.1429179
4: -0.0277857, 0.0573916, -0.0581345, 0.1992887, -0.2270745, 0.1155261

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.28 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.30 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0037549, 0.0340886, -0.0531580, 0.1058602
1: -0.0749285, 0.1370497, -0.0356151, 0.0616529, -0.1365814, 0.1726648
2: -0.0416235, 0.1398997, -0.0105594, 0.0539910, -0.0956145, 0.1504591
3: -0.0797155, 0.1714330, -0.0435620, 0.0632024, -0.1429179, 0.2149950
4: -0.0581345, 0.1992887, -0.0277857, 0.0573916, -0.1155261, 0.2270745

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462205
time: 0.30 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.29 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0190694, 0.1021053, -0.1211746, 0.1211746
1: -0.0749285, 0.1370497, -0.0749285, 0.1370497, -0.2119782, 0.2119782
2: -0.0416235, 0.1398997, -0.0416235, 0.1398997, -0.1815232, 0.1815232
3: -0.0797155, 0.1714330, -0.0797155, 0.1714330, -0.2511485, 0.2511485
4: -0.0581345, 0.1992887, -0.0581345, 0.1992887, -0.2574233, 0.2574233

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462205
time: 0.30 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.29 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.97 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.97
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.97
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462205
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.97
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.97
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462205
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.97
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0037549, 0.0340886, -0.0376604, 0.0373968
1: -0.0350209, 0.0608409, -0.0356151, 0.0616529, -0.0966738, 0.0964560
2: -0.0102656, 0.0532160, -0.0105594, 0.0539910, -0.0642565, 0.0637753
3: -0.0427086, 0.0621664, -0.0435620, 0.0632024, -0.1059110, 0.1057284
4: -0.0273492, 0.0564452, -0.0277857, 0.0573916, -0.0847408, 0.0842309

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447883, upper bound: 0.0465151
time: 0.29 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
time: 0.28 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
time: 0.28 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0190694, 0.1021053, -0.1056770, 0.0527113
1: -0.0350209, 0.0608409, -0.0749285, 0.1370497, -0.1720706, 0.1357694
2: -0.0102656, 0.0532160, -0.0416235, 0.1398997, -0.1501652, 0.0948395
3: -0.0427086, 0.0621664, -0.0797155, 0.1714330, -0.2141416, 0.1418818
4: -0.0273492, 0.0564452, -0.0581345, 0.1992887, -0.2266379, 0.1145797

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447348, upper bound: 0.0465151
time: 0.29 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
time: 0.29 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
time: 0.28 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0037549, 0.0340886, -0.0527285, 0.1038970
1: -0.0738246, 0.1346942, -0.0356151, 0.0616529, -0.1354775, 0.1703093
2: -0.0407216, 0.1377040, -0.0105594, 0.0539910, -0.0947125, 0.1482634
3: -0.0786182, 0.1676345, -0.0435620, 0.0632024, -0.1418206, 0.2111965
4: -0.0572622, 0.1955275, -0.0277857, 0.0573916, -0.1146538, 0.2233132

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
time: 0.29 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
time: 0.29 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0190694, 0.1021053, -0.1207452, 0.1192114
1: -0.0738246, 0.1346942, -0.0749285, 0.1370497, -0.2108742, 0.2096227
2: -0.0407216, 0.1377040, -0.0416235, 0.1398997, -0.1806213, 0.1793275
3: -0.0786182, 0.1676345, -0.0797155, 0.1714330, -0.2500512, 0.2473500
4: -0.0572622, 0.1955275, -0.0581345, 0.1992887, -0.2565510, 0.2536620

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
time: 0.28 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.31 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.89 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
Binary search (step 2): status=Status.VERIFIED, low=0.1590909, high=0.1818182, mid=0.1590909, abs_max=0.05155529826879501
rel_dist={0: [-0.0465725463803796, 0.04657255964042008]}

## Binary search (step 3) starts
Candidate diff: 0.1704546


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
time: 0.29 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.28 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.72 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0070904, 0.0444649, -0.0482198, 0.0411790
1: -0.0356151, 0.0616529, -0.0457317, 0.0752505, -0.1108656, 0.1073847
2: -0.0105594, 0.0539910, -0.0161715, 0.0679093, -0.0784687, 0.0701625
3: -0.0435620, 0.0632024, -0.0558398, 0.0794108, -0.1229728, 0.1190422
4: -0.0277857, 0.0573916, -0.0350063, 0.0803186, -0.1081043, 0.0923979

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
time: 0.28 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.28 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0070904, 0.0444649, -0.0635343, 0.1091957
1: -0.0749285, 0.1370497, -0.0457317, 0.0752505, -0.1501790, 0.1827814
2: -0.0416235, 0.1398997, -0.0161715, 0.0679093, -0.1095328, 0.1560712
3: -0.0797155, 0.1714330, -0.0558398, 0.0794108, -0.1591263, 0.2272728
4: -0.0581345, 0.1992887, -0.0350063, 0.0803186, -0.1384531, 0.2342951

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
time: 0.28 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.22 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0037549, 0.0340886, -0.0378435, 0.0378435
1: -0.0356151, 0.0616529, -0.0356151, 0.0616529, -0.0972680, 0.0972680
2: -0.0105594, 0.0539910, -0.0105594, 0.0539910, -0.0645503, 0.0645503
3: -0.0435620, 0.0632024, -0.0435620, 0.0632024, -0.1067644, 0.1067644
4: -0.0277857, 0.0573916, -0.0277857, 0.0573916, -0.0851773, 0.0851773

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.28 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.29 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0190694, 0.1021053, -0.1058602, 0.0531580
1: -0.0356151, 0.0616529, -0.0749285, 0.1370497, -0.1726648, 0.1365814
2: -0.0105594, 0.0539910, -0.0416235, 0.1398997, -0.1504591, 0.0956145
3: -0.0435620, 0.0632024, -0.0797155, 0.1714330, -0.2149950, 0.1429179
4: -0.0277857, 0.0573916, -0.0581345, 0.1992887, -0.2270745, 0.1155261

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.27 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.28 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0037549, 0.0340886, -0.0531580, 0.1058602
1: -0.0749285, 0.1370497, -0.0356151, 0.0616529, -0.1365814, 0.1726648
2: -0.0416235, 0.1398997, -0.0105594, 0.0539910, -0.0956145, 0.1504591
3: -0.0797155, 0.1714330, -0.0435620, 0.0632024, -0.1429179, 0.2149950
4: -0.0581345, 0.1992887, -0.0277857, 0.0573916, -0.1155261, 0.2270745

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462383
time: 0.29 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.30 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0190694, 0.1021053, -0.1211746, 0.1211746
1: -0.0749285, 0.1370497, -0.0749285, 0.1370497, -0.2119782, 0.2119782
2: -0.0416235, 0.1398997, -0.0416235, 0.1398997, -0.1815232, 0.1815232
3: -0.0797155, 0.1714330, -0.0797155, 0.1714330, -0.2511485, 0.2511485
4: -0.0581345, 0.1992887, -0.0581345, 0.1992887, -0.2574233, 0.2574233

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462383
time: 0.29 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.28 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.67 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.67
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.67
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462383
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.67
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.67
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462383
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.67
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0037549, 0.0340886, -0.0376604, 0.0373968
1: -0.0350209, 0.0608409, -0.0356151, 0.0616529, -0.0966738, 0.0964560
2: -0.0102656, 0.0532160, -0.0105594, 0.0539910, -0.0642565, 0.0637753
3: -0.0427086, 0.0621664, -0.0435620, 0.0632024, -0.1059110, 0.1057284
4: -0.0273492, 0.0564452, -0.0277857, 0.0573916, -0.0847408, 0.0842309

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447883, upper bound: 0.0465151
time: 0.29 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
time: 0.29 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
time: 0.27 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0190694, 0.1021053, -0.1056770, 0.0527113
1: -0.0350209, 0.0608409, -0.0749285, 0.1370497, -0.1720706, 0.1357694
2: -0.0102656, 0.0532160, -0.0416235, 0.1398997, -0.1501652, 0.0948395
3: -0.0427086, 0.0621664, -0.0797155, 0.1714330, -0.2141416, 0.1418818
4: -0.0273492, 0.0564452, -0.0581345, 0.1992887, -0.2266379, 0.1145797

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447348, upper bound: 0.0465151
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
time: 0.27 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
time: 0.29 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0037549, 0.0340886, -0.0527285, 0.1038970
1: -0.0738246, 0.1346942, -0.0356151, 0.0616529, -0.1354775, 0.1703093
2: -0.0407216, 0.1377040, -0.0105594, 0.0539910, -0.0947125, 0.1482634
3: -0.0786182, 0.1676345, -0.0435620, 0.0632024, -0.1418206, 0.2111965
4: -0.0572622, 0.1955275, -0.0277857, 0.0573916, -0.1146538, 0.2233132

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
time: 0.29 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
time: 0.29 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0190694, 0.1021053, -0.1207452, 0.1192114
1: -0.0738246, 0.1346942, -0.0749285, 0.1370497, -0.2108742, 0.2096227
2: -0.0407216, 0.1377040, -0.0416235, 0.1398997, -0.1806213, 0.1793275
3: -0.0786182, 0.1676345, -0.0797155, 0.1714330, -0.2500512, 0.2473500
4: -0.0572622, 0.1955275, -0.0581345, 0.1992887, -0.2565510, 0.2536620

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
time: 0.28 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.30 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.73 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
Binary search (step 3): status=Status.VERIFIED, low=0.1704546, high=0.1818182, mid=0.1704546, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 4) starts
Candidate diff: 0.1761364


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
time: 0.32 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.30 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.79 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0070904, 0.0444649, -0.0482198, 0.0411790
1: -0.0356151, 0.0616529, -0.0457317, 0.0752505, -0.1108656, 0.1073847
2: -0.0105594, 0.0539910, -0.0161715, 0.0679093, -0.0784687, 0.0701625
3: -0.0435620, 0.0632024, -0.0558398, 0.0794108, -0.1229728, 0.1190422
4: -0.0277857, 0.0573916, -0.0350063, 0.0803186, -0.1081043, 0.0923979

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
time: 0.30 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.31 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0070904, 0.0444649, -0.0635343, 0.1091957
1: -0.0749285, 0.1370497, -0.0457317, 0.0752505, -0.1501790, 0.1827814
2: -0.0416235, 0.1398997, -0.0161715, 0.0679093, -0.1095328, 0.1560712
3: -0.0797155, 0.1714330, -0.0558398, 0.0794108, -0.1591263, 0.2272728
4: -0.0581345, 0.1992887, -0.0350063, 0.0803186, -0.1384531, 0.2342951

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
time: 0.29 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.61 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0037549, 0.0340886, -0.0378435, 0.0378435
1: -0.0356151, 0.0616529, -0.0356151, 0.0616529, -0.0972680, 0.0972680
2: -0.0105594, 0.0539910, -0.0105594, 0.0539910, -0.0645503, 0.0645503
3: -0.0435620, 0.0632024, -0.0435620, 0.0632024, -0.1067644, 0.1067644
4: -0.0277857, 0.0573916, -0.0277857, 0.0573916, -0.0851773, 0.0851773

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.28 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.30 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0190694, 0.1021053, -0.1058602, 0.0531580
1: -0.0356151, 0.0616529, -0.0749285, 0.1370497, -0.1726648, 0.1365814
2: -0.0105594, 0.0539910, -0.0416235, 0.1398997, -0.1504591, 0.0956145
3: -0.0435620, 0.0632024, -0.0797155, 0.1714330, -0.2149950, 0.1429179
4: -0.0277857, 0.0573916, -0.0581345, 0.1992887, -0.2270745, 0.1155261

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.28 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.30 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0037549, 0.0340886, -0.0531580, 0.1058602
1: -0.0749285, 0.1370497, -0.0356151, 0.0616529, -0.1365814, 0.1726648
2: -0.0416235, 0.1398997, -0.0105594, 0.0539910, -0.0956145, 0.1504591
3: -0.0797155, 0.1714330, -0.0435620, 0.0632024, -0.1429179, 0.2149950
4: -0.0581345, 0.1992887, -0.0277857, 0.0573916, -0.1155261, 0.2270745

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462459
time: 0.31 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.34 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0190694, 0.1021053, -0.1211746, 0.1211746
1: -0.0749285, 0.1370497, -0.0749285, 0.1370497, -0.2119782, 0.2119782
2: -0.0416235, 0.1398997, -0.0416235, 0.1398997, -0.1815232, 0.1815232
3: -0.0797155, 0.1714330, -0.0797155, 0.1714330, -0.2511485, 0.2511485
4: -0.0581345, 0.1992887, -0.0581345, 0.1992887, -0.2574233, 0.2574233

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462459
time: 0.29 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.28 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.83 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.83
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.83
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462459
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.83
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.83
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462459
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.83
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0037549, 0.0340886, -0.0376604, 0.0373968
1: -0.0350209, 0.0608409, -0.0356151, 0.0616529, -0.0966738, 0.0964560
2: -0.0102656, 0.0532160, -0.0105594, 0.0539910, -0.0642565, 0.0637753
3: -0.0427086, 0.0621664, -0.0435620, 0.0632024, -0.1059110, 0.1057284
4: -0.0273492, 0.0564452, -0.0277857, 0.0573916, -0.0847408, 0.0842309

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447883, upper bound: 0.0465151
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
time: 0.29 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
time: 0.28 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0190694, 0.1021053, -0.1056770, 0.0527113
1: -0.0350209, 0.0608409, -0.0749285, 0.1370497, -0.1720706, 0.1357694
2: -0.0102656, 0.0532160, -0.0416235, 0.1398997, -0.1501652, 0.0948395
3: -0.0427086, 0.0621664, -0.0797155, 0.1714330, -0.2141416, 0.1418818
4: -0.0273492, 0.0564452, -0.0581345, 0.1992887, -0.2266379, 0.1145797

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447348, upper bound: 0.0465151
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
time: 0.29 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
time: 0.27 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0037549, 0.0340886, -0.0527285, 0.1038970
1: -0.0738246, 0.1346942, -0.0356151, 0.0616529, -0.1354775, 0.1703093
2: -0.0407216, 0.1377040, -0.0105594, 0.0539910, -0.0947125, 0.1482634
3: -0.0786182, 0.1676345, -0.0435620, 0.0632024, -0.1418206, 0.2111965
4: -0.0572622, 0.1955275, -0.0277857, 0.0573916, -0.1146538, 0.2233132

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
time: 0.30 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
time: 0.29 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0190694, 0.1021053, -0.1207452, 0.1192114
1: -0.0738246, 0.1346942, -0.0749285, 0.1370497, -0.2108742, 0.2096227
2: -0.0407216, 0.1377040, -0.0416235, 0.1398997, -0.1806213, 0.1793275
3: -0.0786182, 0.1676345, -0.0797155, 0.1714330, -0.2500512, 0.2473500
4: -0.0572622, 0.1955275, -0.0581345, 0.1992887, -0.2565510, 0.2536620

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
time: 0.28 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.28 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.92 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.92
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
Binary search (step 4): status=Status.VERIFIED, low=0.1761364, high=0.1818182, mid=0.1761364, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 5) starts
Candidate diff: 0.1789773


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
time: 0.28 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.28 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.70 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.70
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.70
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0070904, 0.0444649, -0.0482198, 0.0411790
1: -0.0356151, 0.0616529, -0.0457317, 0.0752505, -0.1108656, 0.1073847
2: -0.0105594, 0.0539910, -0.0161715, 0.0679093, -0.0784687, 0.0701625
3: -0.0435620, 0.0632024, -0.0558398, 0.0794108, -0.1229728, 0.1190422
4: -0.0277857, 0.0573916, -0.0350063, 0.0803186, -0.1081043, 0.0923979

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
time: 0.28 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.29 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0070904, 0.0444649, -0.0635343, 0.1091957
1: -0.0749285, 0.1370497, -0.0457317, 0.0752505, -0.1501790, 0.1827814
2: -0.0416235, 0.1398997, -0.0161715, 0.0679093, -0.1095328, 0.1560712
3: -0.0797155, 0.1714330, -0.0558398, 0.0794108, -0.1591263, 0.2272728
4: -0.0581345, 0.1992887, -0.0350063, 0.0803186, -0.1384531, 0.2342951

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
time: 0.28 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.28 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.23 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0037549, 0.0340886, -0.0378435, 0.0378435
1: -0.0356151, 0.0616529, -0.0356151, 0.0616529, -0.0972680, 0.0972680
2: -0.0105594, 0.0539910, -0.0105594, 0.0539910, -0.0645503, 0.0645503
3: -0.0435620, 0.0632024, -0.0435620, 0.0632024, -0.1067644, 0.1067644
4: -0.0277857, 0.0573916, -0.0277857, 0.0573916, -0.0851773, 0.0851773

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.27 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.27 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0190694, 0.1021053, -0.1058602, 0.0531580
1: -0.0356151, 0.0616529, -0.0749285, 0.1370497, -0.1726648, 0.1365814
2: -0.0105594, 0.0539910, -0.0416235, 0.1398997, -0.1504591, 0.0956145
3: -0.0435620, 0.0632024, -0.0797155, 0.1714330, -0.2149950, 0.1429179
4: -0.0277857, 0.0573916, -0.0581345, 0.1992887, -0.2270745, 0.1155261

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.27 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.28 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0037549, 0.0340886, -0.0531580, 0.1058602
1: -0.0749285, 0.1370497, -0.0356151, 0.0616529, -0.1365814, 0.1726648
2: -0.0416235, 0.1398997, -0.0105594, 0.0539910, -0.0956145, 0.1504591
3: -0.0797155, 0.1714330, -0.0435620, 0.0632024, -0.1429179, 0.2149950
4: -0.0581345, 0.1992887, -0.0277857, 0.0573916, -0.1155261, 0.2270745

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462496
time: 0.29 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.34 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0190694, 0.1021053, -0.1211746, 0.1211746
1: -0.0749285, 0.1370497, -0.0749285, 0.1370497, -0.2119782, 0.2119782
2: -0.0416235, 0.1398997, -0.0416235, 0.1398997, -0.1815232, 0.1815232
3: -0.0797155, 0.1714330, -0.0797155, 0.1714330, -0.2511485, 0.2511485
4: -0.0581345, 0.1992887, -0.0581345, 0.1992887, -0.2574233, 0.2574233

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462496
time: 0.28 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.29 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.71 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.71
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.71
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.71
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.71
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.71
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462496
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.71
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.71
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462496
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.71
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0037549, 0.0340886, -0.0376604, 0.0373968
1: -0.0350209, 0.0608409, -0.0356151, 0.0616529, -0.0966738, 0.0964560
2: -0.0102656, 0.0532160, -0.0105594, 0.0539910, -0.0642565, 0.0637753
3: -0.0427086, 0.0621664, -0.0435620, 0.0632024, -0.1059110, 0.1057284
4: -0.0273492, 0.0564452, -0.0277857, 0.0573916, -0.0847408, 0.0842309

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447883, upper bound: 0.0465151
time: 0.27 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
time: 0.27 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
time: 0.26 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0190694, 0.1021053, -0.1056770, 0.0527113
1: -0.0350209, 0.0608409, -0.0749285, 0.1370497, -0.1720706, 0.1357694
2: -0.0102656, 0.0532160, -0.0416235, 0.1398997, -0.1501652, 0.0948395
3: -0.0427086, 0.0621664, -0.0797155, 0.1714330, -0.2141416, 0.1418818
4: -0.0273492, 0.0564452, -0.0581345, 0.1992887, -0.2266379, 0.1145797

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447348, upper bound: 0.0465151
time: 0.30 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
time: 0.29 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
time: 0.26 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0037549, 0.0340886, -0.0527285, 0.1038970
1: -0.0738246, 0.1346942, -0.0356151, 0.0616529, -0.1354775, 0.1703093
2: -0.0407216, 0.1377040, -0.0105594, 0.0539910, -0.0947125, 0.1482634
3: -0.0786182, 0.1676345, -0.0435620, 0.0632024, -0.1418206, 0.2111965
4: -0.0572622, 0.1955275, -0.0277857, 0.0573916, -0.1146538, 0.2233132

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
time: 0.29 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
time: 0.28 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0190694, 0.1021053, -0.1207452, 0.1192114
1: -0.0738246, 0.1346942, -0.0749285, 0.1370497, -0.2108742, 0.2096227
2: -0.0407216, 0.1377040, -0.0416235, 0.1398997, -0.1806213, 0.1793275
3: -0.0786182, 0.1676345, -0.0797155, 0.1714330, -0.2500512, 0.2473500
4: -0.0572622, 0.1955275, -0.0581345, 0.1992887, -0.2565510, 0.2536620

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
time: 0.27 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.28 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.58 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
Binary search (step 5): status=Status.VERIFIED, low=0.1789773, high=0.1818182, mid=0.1789773, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 6) starts
Candidate diff: 0.1803977


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
time: 0.30 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.29 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.75 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.75
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0070904, 0.0444649, -0.0482198, 0.0411790
1: -0.0356151, 0.0616529, -0.0457317, 0.0752505, -0.1108656, 0.1073847
2: -0.0105594, 0.0539910, -0.0161715, 0.0679093, -0.0784687, 0.0701625
3: -0.0435620, 0.0632024, -0.0558398, 0.0794108, -0.1229728, 0.1190422
4: -0.0277857, 0.0573916, -0.0350063, 0.0803186, -0.1081043, 0.0923979

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
time: 0.30 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.29 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0070904, 0.0444649, -0.0635343, 0.1091957
1: -0.0749285, 0.1370497, -0.0457317, 0.0752505, -0.1501790, 0.1827814
2: -0.0416235, 0.1398997, -0.0161715, 0.0679093, -0.1095328, 0.1560712
3: -0.0797155, 0.1714330, -0.0558398, 0.0794108, -0.1591263, 0.2272728
4: -0.0581345, 0.1992887, -0.0350063, 0.0803186, -0.1384531, 0.2342951

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
time: 0.29 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.52 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.52
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.52
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.52
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.52
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0037549, 0.0340886, -0.0378435, 0.0378435
1: -0.0356151, 0.0616529, -0.0356151, 0.0616529, -0.0972680, 0.0972680
2: -0.0105594, 0.0539910, -0.0105594, 0.0539910, -0.0645503, 0.0645503
3: -0.0435620, 0.0632024, -0.0435620, 0.0632024, -0.1067644, 0.1067644
4: -0.0277857, 0.0573916, -0.0277857, 0.0573916, -0.0851773, 0.0851773

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.28 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.28 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0190694, 0.1021053, -0.1058602, 0.0531580
1: -0.0356151, 0.0616529, -0.0749285, 0.1370497, -0.1726648, 0.1365814
2: -0.0105594, 0.0539910, -0.0416235, 0.1398997, -0.1504591, 0.0956145
3: -0.0435620, 0.0632024, -0.0797155, 0.1714330, -0.2149950, 0.1429179
4: -0.0277857, 0.0573916, -0.0581345, 0.1992887, -0.2270745, 0.1155261

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.30 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.31 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0037549, 0.0340886, -0.0531580, 0.1058602
1: -0.0749285, 0.1370497, -0.0356151, 0.0616529, -0.1365814, 0.1726648
2: -0.0416235, 0.1398997, -0.0105594, 0.0539910, -0.0956145, 0.1504591
3: -0.0797155, 0.1714330, -0.0435620, 0.0632024, -0.1429179, 0.2149950
4: -0.0581345, 0.1992887, -0.0277857, 0.0573916, -0.1155261, 0.2270745

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462515
time: 0.28 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0190694, 0.1021053, -0.1211746, 0.1211746
1: -0.0749285, 0.1370497, -0.0749285, 0.1370497, -0.2119782, 0.2119782
2: -0.0416235, 0.1398997, -0.0416235, 0.1398997, -0.1815232, 0.1815232
3: -0.0797155, 0.1714330, -0.0797155, 0.1714330, -0.2511485, 0.2511485
4: -0.0581345, 0.1992887, -0.0581345, 0.1992887, -0.2574233, 0.2574233

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462515
time: 0.28 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.28 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.85 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.85
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.85
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462515
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.85
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.85
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462515
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.85
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0037549, 0.0340886, -0.0376604, 0.0373968
1: -0.0350209, 0.0608409, -0.0356151, 0.0616529, -0.0966738, 0.0964560
2: -0.0102656, 0.0532160, -0.0105594, 0.0539910, -0.0642565, 0.0637753
3: -0.0427086, 0.0621664, -0.0435620, 0.0632024, -0.1059110, 0.1057284
4: -0.0273492, 0.0564452, -0.0277857, 0.0573916, -0.0847408, 0.0842309

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447883, upper bound: 0.0465151
time: 0.28 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
time: 0.29 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
time: 0.28 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0190694, 0.1021053, -0.1056770, 0.0527113
1: -0.0350209, 0.0608409, -0.0749285, 0.1370497, -0.1720706, 0.1357694
2: -0.0102656, 0.0532160, -0.0416235, 0.1398997, -0.1501652, 0.0948395
3: -0.0427086, 0.0621664, -0.0797155, 0.1714330, -0.2141416, 0.1418818
4: -0.0273492, 0.0564452, -0.0581345, 0.1992887, -0.2266379, 0.1145797

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447348, upper bound: 0.0465151
time: 0.29 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
time: 0.29 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
time: 0.29 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0037549, 0.0340886, -0.0527285, 0.1038970
1: -0.0738246, 0.1346942, -0.0356151, 0.0616529, -0.1354775, 0.1703093
2: -0.0407216, 0.1377040, -0.0105594, 0.0539910, -0.0947125, 0.1482634
3: -0.0786182, 0.1676345, -0.0435620, 0.0632024, -0.1418206, 0.2111965
4: -0.0572622, 0.1955275, -0.0277857, 0.0573916, -0.1146538, 0.2233132

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
time: 0.29 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
time: 0.29 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0190694, 0.1021053, -0.1207452, 0.1192114
1: -0.0738246, 0.1346942, -0.0749285, 0.1370497, -0.2108742, 0.2096227
2: -0.0407216, 0.1377040, -0.0416235, 0.1398997, -0.1806213, 0.1793275
3: -0.0786182, 0.1676345, -0.0797155, 0.1714330, -0.2500512, 0.2473500
4: -0.0572622, 0.1955275, -0.0581345, 0.1992887, -0.2565510, 0.2536620

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
time: 0.28 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.28 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.63 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.63
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
Binary search (step 6): status=Status.VERIFIED, low=0.1803977, high=0.1818182, mid=0.1803977, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 7) starts
Candidate diff: 0.1811080


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
time: 0.29 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.27 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.70 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.70
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.70
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0070904, 0.0444649, -0.0482198, 0.0411790
1: -0.0356151, 0.0616529, -0.0457317, 0.0752505, -0.1108656, 0.1073847
2: -0.0105594, 0.0539910, -0.0161715, 0.0679093, -0.0784687, 0.0701625
3: -0.0435620, 0.0632024, -0.0558398, 0.0794108, -0.1229728, 0.1190422
4: -0.0277857, 0.0573916, -0.0350063, 0.0803186, -0.1081043, 0.0923979

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
time: 0.27 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.27 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0070904, 0.0444649, -0.0635343, 0.1091957
1: -0.0749285, 0.1370497, -0.0457317, 0.0752505, -0.1501790, 0.1827814
2: -0.0416235, 0.1398997, -0.0161715, 0.0679093, -0.1095328, 0.1560712
3: -0.0797155, 0.1714330, -0.0558398, 0.0794108, -0.1591263, 0.2272728
4: -0.0581345, 0.1992887, -0.0350063, 0.0803186, -0.1384531, 0.2342951

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
time: 0.28 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.28 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.20 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0037549, 0.0340886, -0.0378435, 0.0378435
1: -0.0356151, 0.0616529, -0.0356151, 0.0616529, -0.0972680, 0.0972680
2: -0.0105594, 0.0539910, -0.0105594, 0.0539910, -0.0645503, 0.0645503
3: -0.0435620, 0.0632024, -0.0435620, 0.0632024, -0.1067644, 0.1067644
4: -0.0277857, 0.0573916, -0.0277857, 0.0573916, -0.0851773, 0.0851773

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.27 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.28 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0190694, 0.1021053, -0.1058602, 0.0531580
1: -0.0356151, 0.0616529, -0.0749285, 0.1370497, -0.1726648, 0.1365814
2: -0.0105594, 0.0539910, -0.0416235, 0.1398997, -0.1504591, 0.0956145
3: -0.0435620, 0.0632024, -0.0797155, 0.1714330, -0.2149950, 0.1429179
4: -0.0277857, 0.0573916, -0.0581345, 0.1992887, -0.2270745, 0.1155261

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.27 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.28 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0037549, 0.0340886, -0.0531580, 0.1058602
1: -0.0749285, 0.1370497, -0.0356151, 0.0616529, -0.1365814, 0.1726648
2: -0.0416235, 0.1398997, -0.0105594, 0.0539910, -0.0956145, 0.1504591
3: -0.0797155, 0.1714330, -0.0435620, 0.0632024, -0.1429179, 0.2149950
4: -0.0581345, 0.1992887, -0.0277857, 0.0573916, -0.1155261, 0.2270745

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462524
time: 0.28 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0190694, 0.1021053, -0.1211746, 0.1211746
1: -0.0749285, 0.1370497, -0.0749285, 0.1370497, -0.2119782, 0.2119782
2: -0.0416235, 0.1398997, -0.0416235, 0.1398997, -0.1815232, 0.1815232
3: -0.0797155, 0.1714330, -0.0797155, 0.1714330, -0.2511485, 0.2511485
4: -0.0581345, 0.1992887, -0.0581345, 0.1992887, -0.2574233, 0.2574233

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462524
time: 0.28 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.28 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.53 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462524
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462524
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0037549, 0.0340886, -0.0376604, 0.0373968
1: -0.0350209, 0.0608409, -0.0356151, 0.0616529, -0.0966738, 0.0964560
2: -0.0102656, 0.0532160, -0.0105594, 0.0539910, -0.0642565, 0.0637753
3: -0.0427086, 0.0621664, -0.0435620, 0.0632024, -0.1059110, 0.1057284
4: -0.0273492, 0.0564452, -0.0277857, 0.0573916, -0.0847408, 0.0842309

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447883, upper bound: 0.0465151
time: 0.29 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
time: 0.28 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
time: 0.26 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0190694, 0.1021053, -0.1056770, 0.0527113
1: -0.0350209, 0.0608409, -0.0749285, 0.1370497, -0.1720706, 0.1357694
2: -0.0102656, 0.0532160, -0.0416235, 0.1398997, -0.1501652, 0.0948395
3: -0.0427086, 0.0621664, -0.0797155, 0.1714330, -0.2141416, 0.1418818
4: -0.0273492, 0.0564452, -0.0581345, 0.1992887, -0.2266379, 0.1145797

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447348, upper bound: 0.0465151
time: 0.30 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
time: 0.28 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
time: 0.26 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0037549, 0.0340886, -0.0527285, 0.1038970
1: -0.0738246, 0.1346942, -0.0356151, 0.0616529, -0.1354775, 0.1703093
2: -0.0407216, 0.1377040, -0.0105594, 0.0539910, -0.0947125, 0.1482634
3: -0.0786182, 0.1676345, -0.0435620, 0.0632024, -0.1418206, 0.2111965
4: -0.0572622, 0.1955275, -0.0277857, 0.0573916, -0.1146538, 0.2233132

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
time: 0.28 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
time: 0.27 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0190694, 0.1021053, -0.1207452, 0.1192114
1: -0.0738246, 0.1346942, -0.0749285, 0.1370497, -0.2108742, 0.2096227
2: -0.0407216, 0.1377040, -0.0416235, 0.1398997, -0.1806213, 0.1793275
3: -0.0786182, 0.1676345, -0.0797155, 0.1714330, -0.2500512, 0.2473500
4: -0.0572622, 0.1955275, -0.0581345, 0.1992887, -0.2565510, 0.2536620

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
time: 0.28 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.30 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.59 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
Binary search (step 7): status=Status.VERIFIED, low=0.1811080, high=0.1818182, mid=0.1811080, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 8) starts
Candidate diff: 0.1814631


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
time: 0.30 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.29 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.74 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.74
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.74
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0070904, 0.0444649, -0.0482198, 0.0411790
1: -0.0356151, 0.0616529, -0.0457317, 0.0752505, -0.1108656, 0.1073847
2: -0.0105594, 0.0539910, -0.0161715, 0.0679093, -0.0784687, 0.0701625
3: -0.0435620, 0.0632024, -0.0558398, 0.0794108, -0.1229728, 0.1190422
4: -0.0277857, 0.0573916, -0.0350063, 0.0803186, -0.1081043, 0.0923979

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
time: 0.28 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.29 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0070904, 0.0444649, -0.0635343, 0.1091957
1: -0.0749285, 0.1370497, -0.0457317, 0.0752505, -0.1501790, 0.1827814
2: -0.0416235, 0.1398997, -0.0161715, 0.0679093, -0.1095328, 0.1560712
3: -0.0797155, 0.1714330, -0.0558398, 0.0794108, -0.1591263, 0.2272728
4: -0.0581345, 0.1992887, -0.0350063, 0.0803186, -0.1384531, 0.2342951

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
time: 0.30 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.50 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.50
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0037549, 0.0340886, -0.0378435, 0.0378435
1: -0.0356151, 0.0616529, -0.0356151, 0.0616529, -0.0972680, 0.0972680
2: -0.0105594, 0.0539910, -0.0105594, 0.0539910, -0.0645503, 0.0645503
3: -0.0435620, 0.0632024, -0.0435620, 0.0632024, -0.1067644, 0.1067644
4: -0.0277857, 0.0573916, -0.0277857, 0.0573916, -0.0851773, 0.0851773

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.27 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.28 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0190694, 0.1021053, -0.1058602, 0.0531580
1: -0.0356151, 0.0616529, -0.0749285, 0.1370497, -0.1726648, 0.1365814
2: -0.0105594, 0.0539910, -0.0416235, 0.1398997, -0.1504591, 0.0956145
3: -0.0435620, 0.0632024, -0.0797155, 0.1714330, -0.2149950, 0.1429179
4: -0.0277857, 0.0573916, -0.0581345, 0.1992887, -0.2270745, 0.1155261

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.29 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.29 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0037549, 0.0340886, -0.0531580, 0.1058602
1: -0.0749285, 0.1370497, -0.0356151, 0.0616529, -0.1365814, 0.1726648
2: -0.0416235, 0.1398997, -0.0105594, 0.0539910, -0.0956145, 0.1504591
3: -0.0797155, 0.1714330, -0.0435620, 0.0632024, -0.1429179, 0.2149950
4: -0.0581345, 0.1992887, -0.0277857, 0.0573916, -0.1155261, 0.2270745

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462529
time: 0.29 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0190694, 0.1021053, -0.1211746, 0.1211746
1: -0.0749285, 0.1370497, -0.0749285, 0.1370497, -0.2119782, 0.2119782
2: -0.0416235, 0.1398997, -0.0416235, 0.1398997, -0.1815232, 0.1815232
3: -0.0797155, 0.1714330, -0.0797155, 0.1714330, -0.2511485, 0.2511485
4: -0.0581345, 0.1992887, -0.0581345, 0.1992887, -0.2574233, 0.2574233

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462529
time: 0.28 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.28 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.80 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.80
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.80
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462529
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.80
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.80
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462529
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.80
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0037549, 0.0340886, -0.0376604, 0.0373968
1: -0.0350209, 0.0608409, -0.0356151, 0.0616529, -0.0966738, 0.0964560
2: -0.0102656, 0.0532160, -0.0105594, 0.0539910, -0.0642565, 0.0637753
3: -0.0427086, 0.0621664, -0.0435620, 0.0632024, -0.1059110, 0.1057284
4: -0.0273492, 0.0564452, -0.0277857, 0.0573916, -0.0847408, 0.0842309

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447883, upper bound: 0.0465151
time: 0.28 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
time: 0.29 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
time: 0.28 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0190694, 0.1021053, -0.1056770, 0.0527113
1: -0.0350209, 0.0608409, -0.0749285, 0.1370497, -0.1720706, 0.1357694
2: -0.0102656, 0.0532160, -0.0416235, 0.1398997, -0.1501652, 0.0948395
3: -0.0427086, 0.0621664, -0.0797155, 0.1714330, -0.2141416, 0.1418818
4: -0.0273492, 0.0564452, -0.0581345, 0.1992887, -0.2266379, 0.1145797

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447348, upper bound: 0.0465151
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
time: 0.29 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
time: 0.27 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0037549, 0.0340886, -0.0527285, 0.1038970
1: -0.0738246, 0.1346942, -0.0356151, 0.0616529, -0.1354775, 0.1703093
2: -0.0407216, 0.1377040, -0.0105594, 0.0539910, -0.0947125, 0.1482634
3: -0.0786182, 0.1676345, -0.0435620, 0.0632024, -0.1418206, 0.2111965
4: -0.0572622, 0.1955275, -0.0277857, 0.0573916, -0.1146538, 0.2233132

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
time: 0.29 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
time: 0.29 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0190694, 0.1021053, -0.1207452, 0.1192114
1: -0.0738246, 0.1346942, -0.0749285, 0.1370497, -0.2108742, 0.2096227
2: -0.0407216, 0.1377040, -0.0416235, 0.1398997, -0.1806213, 0.1793275
3: -0.0786182, 0.1676345, -0.0797155, 0.1714330, -0.2500512, 0.2473500
4: -0.0572622, 0.1955275, -0.0581345, 0.1992887, -0.2565510, 0.2536620

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
time: 0.30 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.28 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.79 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.79
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.79
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.79
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.79
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.79
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.79
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.79
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.79
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
Binary search (step 8): status=Status.VERIFIED, low=0.1814631, high=0.1818182, mid=0.1814631, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 9) starts
Candidate diff: 0.1816406


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
time: 0.29 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.26 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.68 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0070904, 0.0444649, -0.0482198, 0.0411790
1: -0.0356151, 0.0616529, -0.0457317, 0.0752505, -0.1108656, 0.1073847
2: -0.0105594, 0.0539910, -0.0161715, 0.0679093, -0.0784687, 0.0701625
3: -0.0435620, 0.0632024, -0.0558398, 0.0794108, -0.1229728, 0.1190422
4: -0.0277857, 0.0573916, -0.0350063, 0.0803186, -0.1081043, 0.0923979

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
time: 0.27 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.27 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0070904, 0.0444649, -0.0635343, 0.1091957
1: -0.0749285, 0.1370497, -0.0457317, 0.0752505, -0.1501790, 0.1827814
2: -0.0416235, 0.1398997, -0.0161715, 0.0679093, -0.1095328, 0.1560712
3: -0.0797155, 0.1714330, -0.0558398, 0.0794108, -0.1591263, 0.2272728
4: -0.0581345, 0.1992887, -0.0350063, 0.0803186, -0.1384531, 0.2342951

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
time: 0.27 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.18 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0037549, 0.0340886, -0.0378435, 0.0378435
1: -0.0356151, 0.0616529, -0.0356151, 0.0616529, -0.0972680, 0.0972680
2: -0.0105594, 0.0539910, -0.0105594, 0.0539910, -0.0645503, 0.0645503
3: -0.0435620, 0.0632024, -0.0435620, 0.0632024, -0.1067644, 0.1067644
4: -0.0277857, 0.0573916, -0.0277857, 0.0573916, -0.0851773, 0.0851773

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.27 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.27 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0190694, 0.1021053, -0.1058602, 0.0531580
1: -0.0356151, 0.0616529, -0.0749285, 0.1370497, -0.1726648, 0.1365814
2: -0.0105594, 0.0539910, -0.0416235, 0.1398997, -0.1504591, 0.0956145
3: -0.0435620, 0.0632024, -0.0797155, 0.1714330, -0.2149950, 0.1429179
4: -0.0277857, 0.0573916, -0.0581345, 0.1992887, -0.2270745, 0.1155261

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.26 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.28 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0037549, 0.0340886, -0.0531580, 0.1058602
1: -0.0749285, 0.1370497, -0.0356151, 0.0616529, -0.1365814, 0.1726648
2: -0.0416235, 0.1398997, -0.0105594, 0.0539910, -0.0956145, 0.1504591
3: -0.0797155, 0.1714330, -0.0435620, 0.0632024, -0.1429179, 0.2149950
4: -0.0581345, 0.1992887, -0.0277857, 0.0573916, -0.1155261, 0.2270745

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462531
time: 0.29 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0190694, 0.1021053, -0.1211746, 0.1211746
1: -0.0749285, 0.1370497, -0.0749285, 0.1370497, -0.2119782, 0.2119782
2: -0.0416235, 0.1398997, -0.0416235, 0.1398997, -0.1815232, 0.1815232
3: -0.0797155, 0.1714330, -0.0797155, 0.1714330, -0.2511485, 0.2511485
4: -0.0581345, 0.1992887, -0.0581345, 0.1992887, -0.2574233, 0.2574233

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462531
time: 0.27 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.27 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.51 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.51
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.51
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462531
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.51
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.51
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462531
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.51
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0037549, 0.0340886, -0.0376604, 0.0373968
1: -0.0350209, 0.0608409, -0.0356151, 0.0616529, -0.0966738, 0.0964560
2: -0.0102656, 0.0532160, -0.0105594, 0.0539910, -0.0642565, 0.0637753
3: -0.0427086, 0.0621664, -0.0435620, 0.0632024, -0.1059110, 0.1057284
4: -0.0273492, 0.0564452, -0.0277857, 0.0573916, -0.0847408, 0.0842309

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447883, upper bound: 0.0465151
time: 0.27 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
time: 0.27 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
time: 0.29 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0190694, 0.1021053, -0.1056770, 0.0527113
1: -0.0350209, 0.0608409, -0.0749285, 0.1370497, -0.1720706, 0.1357694
2: -0.0102656, 0.0532160, -0.0416235, 0.1398997, -0.1501652, 0.0948395
3: -0.0427086, 0.0621664, -0.0797155, 0.1714330, -0.2141416, 0.1418818
4: -0.0273492, 0.0564452, -0.0581345, 0.1992887, -0.2266379, 0.1145797

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447348, upper bound: 0.0465151
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
time: 0.27 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
time: 0.26 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0037549, 0.0340886, -0.0527285, 0.1038970
1: -0.0738246, 0.1346942, -0.0356151, 0.0616529, -0.1354775, 0.1703093
2: -0.0407216, 0.1377040, -0.0105594, 0.0539910, -0.0947125, 0.1482634
3: -0.0786182, 0.1676345, -0.0435620, 0.0632024, -0.1418206, 0.2111965
4: -0.0572622, 0.1955275, -0.0277857, 0.0573916, -0.1146538, 0.2233132

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
time: 0.28 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
time: 0.27 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0190694, 0.1021053, -0.1207452, 0.1192114
1: -0.0738246, 0.1346942, -0.0749285, 0.1370497, -0.2108742, 0.2096227
2: -0.0407216, 0.1377040, -0.0416235, 0.1398997, -0.1806213, 0.1793275
3: -0.0786182, 0.1676345, -0.0797155, 0.1714330, -0.2500512, 0.2473500
4: -0.0572622, 0.1955275, -0.0581345, 0.1992887, -0.2565510, 0.2536620

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
time: 0.29 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.29 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.60 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.60
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
Binary search (step 9): status=Status.VERIFIED, low=0.1816406, high=0.1818182, mid=0.1816406, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 10) starts
Candidate diff: 0.1817294


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
time: 0.29 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.29 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.74 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.74
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.74
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0070904, 0.0444649, -0.0482198, 0.0411790
1: -0.0356151, 0.0616529, -0.0457317, 0.0752505, -0.1108656, 0.1073847
2: -0.0105594, 0.0539910, -0.0161715, 0.0679093, -0.0784687, 0.0701625
3: -0.0435620, 0.0632024, -0.0558398, 0.0794108, -0.1229728, 0.1190422
4: -0.0277857, 0.0573916, -0.0350063, 0.0803186, -0.1081043, 0.0923979

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
time: 0.29 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.29 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0070904, 0.0444649, -0.0635343, 0.1091957
1: -0.0749285, 0.1370497, -0.0457317, 0.0752505, -0.1501790, 0.1827814
2: -0.0416235, 0.1398997, -0.0161715, 0.0679093, -0.1095328, 0.1560712
3: -0.0797155, 0.1714330, -0.0558398, 0.0794108, -0.1591263, 0.2272728
4: -0.0581345, 0.1992887, -0.0350063, 0.0803186, -0.1384531, 0.2342951

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
time: 0.28 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.45 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0037549, 0.0340886, -0.0378435, 0.0378435
1: -0.0356151, 0.0616529, -0.0356151, 0.0616529, -0.0972680, 0.0972680
2: -0.0105594, 0.0539910, -0.0105594, 0.0539910, -0.0645503, 0.0645503
3: -0.0435620, 0.0632024, -0.0435620, 0.0632024, -0.1067644, 0.1067644
4: -0.0277857, 0.0573916, -0.0277857, 0.0573916, -0.0851773, 0.0851773

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.27 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.28 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0190694, 0.1021053, -0.1058602, 0.0531580
1: -0.0356151, 0.0616529, -0.0749285, 0.1370497, -0.1726648, 0.1365814
2: -0.0105594, 0.0539910, -0.0416235, 0.1398997, -0.1504591, 0.0956145
3: -0.0435620, 0.0632024, -0.0797155, 0.1714330, -0.2149950, 0.1429179
4: -0.0277857, 0.0573916, -0.0581345, 0.1992887, -0.2270745, 0.1155261

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.27 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.29 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0037549, 0.0340886, -0.0531580, 0.1058602
1: -0.0749285, 0.1370497, -0.0356151, 0.0616529, -0.1365814, 0.1726648
2: -0.0416235, 0.1398997, -0.0105594, 0.0539910, -0.0956145, 0.1504591
3: -0.0797155, 0.1714330, -0.0435620, 0.0632024, -0.1429179, 0.2149950
4: -0.0581345, 0.1992887, -0.0277857, 0.0573916, -0.1155261, 0.2270745

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462532
time: 0.29 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.34 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0190694, 0.1021053, -0.1211746, 0.1211746
1: -0.0749285, 0.1370497, -0.0749285, 0.1370497, -0.2119782, 0.2119782
2: -0.0416235, 0.1398997, -0.0416235, 0.1398997, -0.1815232, 0.1815232
3: -0.0797155, 0.1714330, -0.0797155, 0.1714330, -0.2511485, 0.2511485
4: -0.0581345, 0.1992887, -0.0581345, 0.1992887, -0.2574233, 0.2574233

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462532
time: 0.31 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.29 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.88 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.88
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.88
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462532
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.88
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.88
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462532
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.88
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0037549, 0.0340886, -0.0376604, 0.0373968
1: -0.0350209, 0.0608409, -0.0356151, 0.0616529, -0.0966738, 0.0964560
2: -0.0102656, 0.0532160, -0.0105594, 0.0539910, -0.0642565, 0.0637753
3: -0.0427086, 0.0621664, -0.0435620, 0.0632024, -0.1059110, 0.1057284
4: -0.0273492, 0.0564452, -0.0277857, 0.0573916, -0.0847408, 0.0842309

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447883, upper bound: 0.0465151
time: 0.28 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
time: 0.29 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
time: 0.30 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0190694, 0.1021053, -0.1056770, 0.0527113
1: -0.0350209, 0.0608409, -0.0749285, 0.1370497, -0.1720706, 0.1357694
2: -0.0102656, 0.0532160, -0.0416235, 0.1398997, -0.1501652, 0.0948395
3: -0.0427086, 0.0621664, -0.0797155, 0.1714330, -0.2141416, 0.1418818
4: -0.0273492, 0.0564452, -0.0581345, 0.1992887, -0.2266379, 0.1145797

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447348, upper bound: 0.0465151
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
time: 0.30 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
time: 0.27 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0037549, 0.0340886, -0.0527285, 0.1038970
1: -0.0738246, 0.1346942, -0.0356151, 0.0616529, -0.1354775, 0.1703093
2: -0.0407216, 0.1377040, -0.0105594, 0.0539910, -0.0947125, 0.1482634
3: -0.0786182, 0.1676345, -0.0435620, 0.0632024, -0.1418206, 0.2111965
4: -0.0572622, 0.1955275, -0.0277857, 0.0573916, -0.1146538, 0.2233132

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
time: 0.29 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
time: 0.29 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0190694, 0.1021053, -0.1207452, 0.1192114
1: -0.0738246, 0.1346942, -0.0749285, 0.1370497, -0.2108742, 0.2096227
2: -0.0407216, 0.1377040, -0.0416235, 0.1398997, -0.1806213, 0.1793275
3: -0.0786182, 0.1676345, -0.0797155, 0.1714330, -0.2500512, 0.2473500
4: -0.0572622, 0.1955275, -0.0581345, 0.1992887, -0.2565510, 0.2536620

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
time: 0.27 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.28 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.68 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
Binary search (step 10): status=Status.VERIFIED, low=0.1817294, high=0.1818182, mid=0.1817294, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 11) starts
Candidate diff: 0.1817738


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
time: 0.28 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.28 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.70 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.70
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.70
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0070904, 0.0444649, -0.0482198, 0.0411790
1: -0.0356151, 0.0616529, -0.0457317, 0.0752505, -0.1108656, 0.1073847
2: -0.0105594, 0.0539910, -0.0161715, 0.0679093, -0.0784687, 0.0701625
3: -0.0435620, 0.0632024, -0.0558398, 0.0794108, -0.1229728, 0.1190422
4: -0.0277857, 0.0573916, -0.0350063, 0.0803186, -0.1081043, 0.0923979

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
time: 0.27 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.27 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0070904, 0.0444649, -0.0635343, 0.1091957
1: -0.0749285, 0.1370497, -0.0457317, 0.0752505, -0.1501790, 0.1827814
2: -0.0416235, 0.1398997, -0.0161715, 0.0679093, -0.1095328, 0.1560712
3: -0.0797155, 0.1714330, -0.0558398, 0.0794108, -0.1591263, 0.2272728
4: -0.0581345, 0.1992887, -0.0350063, 0.0803186, -0.1384531, 0.2342951

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
time: 0.27 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.28 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.18 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.18
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0037549, 0.0340886, -0.0378435, 0.0378435
1: -0.0356151, 0.0616529, -0.0356151, 0.0616529, -0.0972680, 0.0972680
2: -0.0105594, 0.0539910, -0.0105594, 0.0539910, -0.0645503, 0.0645503
3: -0.0435620, 0.0632024, -0.0435620, 0.0632024, -0.1067644, 0.1067644
4: -0.0277857, 0.0573916, -0.0277857, 0.0573916, -0.0851773, 0.0851773

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.27 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.27 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0190694, 0.1021053, -0.1058602, 0.0531580
1: -0.0356151, 0.0616529, -0.0749285, 0.1370497, -0.1726648, 0.1365814
2: -0.0105594, 0.0539910, -0.0416235, 0.1398997, -0.1504591, 0.0956145
3: -0.0435620, 0.0632024, -0.0797155, 0.1714330, -0.2149950, 0.1429179
4: -0.0277857, 0.0573916, -0.0581345, 0.1992887, -0.2270745, 0.1155261

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.28 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.26 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0037549, 0.0340886, -0.0531580, 0.1058602
1: -0.0749285, 0.1370497, -0.0356151, 0.0616529, -0.1365814, 0.1726648
2: -0.0416235, 0.1398997, -0.0105594, 0.0539910, -0.0956145, 0.1504591
3: -0.0797155, 0.1714330, -0.0435620, 0.0632024, -0.1429179, 0.2149950
4: -0.0581345, 0.1992887, -0.0277857, 0.0573916, -0.1155261, 0.2270745

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
time: 0.28 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0190694, 0.1021053, -0.1211746, 0.1211746
1: -0.0749285, 0.1370497, -0.0749285, 0.1370497, -0.2119782, 0.2119782
2: -0.0416235, 0.1398997, -0.0416235, 0.1398997, -0.1815232, 0.1815232
3: -0.0797155, 0.1714330, -0.0797155, 0.1714330, -0.2511485, 0.2511485
4: -0.0581345, 0.1992887, -0.0581345, 0.1992887, -0.2574233, 0.2574233

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
time: 0.29 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.27 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.52 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0037549, 0.0340886, -0.0376604, 0.0373968
1: -0.0350209, 0.0608409, -0.0356151, 0.0616529, -0.0966738, 0.0964560
2: -0.0102656, 0.0532160, -0.0105594, 0.0539910, -0.0642565, 0.0637753
3: -0.0427086, 0.0621664, -0.0435620, 0.0632024, -0.1059110, 0.1057284
4: -0.0273492, 0.0564452, -0.0277857, 0.0573916, -0.0847408, 0.0842309

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447883, upper bound: 0.0465151
time: 0.27 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
time: 0.27 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
time: 0.29 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0190694, 0.1021053, -0.1056770, 0.0527113
1: -0.0350209, 0.0608409, -0.0749285, 0.1370497, -0.1720706, 0.1357694
2: -0.0102656, 0.0532160, -0.0416235, 0.1398997, -0.1501652, 0.0948395
3: -0.0427086, 0.0621664, -0.0797155, 0.1714330, -0.2141416, 0.1418818
4: -0.0273492, 0.0564452, -0.0581345, 0.1992887, -0.2266379, 0.1145797

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447348, upper bound: 0.0465151
time: 0.27 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
time: 0.27 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
time: 0.26 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0037549, 0.0340886, -0.0527285, 0.1038970
1: -0.0738246, 0.1346942, -0.0356151, 0.0616529, -0.1354775, 0.1703093
2: -0.0407216, 0.1377040, -0.0105594, 0.0539910, -0.0947125, 0.1482634
3: -0.0786182, 0.1676345, -0.0435620, 0.0632024, -0.1418206, 0.2111965
4: -0.0572622, 0.1955275, -0.0277857, 0.0573916, -0.1146538, 0.2233132

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
time: 0.28 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
time: 0.28 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0190694, 0.1021053, -0.1207452, 0.1192114
1: -0.0738246, 0.1346942, -0.0749285, 0.1370497, -0.2108742, 0.2096227
2: -0.0407216, 0.1377040, -0.0416235, 0.1398997, -0.1806213, 0.1793275
3: -0.0786182, 0.1676345, -0.0797155, 0.1714330, -0.2500512, 0.2473500
4: -0.0572622, 0.1955275, -0.0581345, 0.1992887, -0.2565510, 0.2536620

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
time: 0.27 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.29 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.56 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
Binary search (step 11): status=Status.VERIFIED, low=0.1817738, high=0.1818182, mid=0.1817738, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 12) starts
Candidate diff: 0.1817960


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
time: 0.29 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.27 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.73 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.73
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0070904, 0.0444649, -0.0482198, 0.0411790
1: -0.0356151, 0.0616529, -0.0457317, 0.0752505, -0.1108656, 0.1073847
2: -0.0105594, 0.0539910, -0.0161715, 0.0679093, -0.0784687, 0.0701625
3: -0.0435620, 0.0632024, -0.0558398, 0.0794108, -0.1229728, 0.1190422
4: -0.0277857, 0.0573916, -0.0350063, 0.0803186, -0.1081043, 0.0923979

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
time: 0.29 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.29 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0070904, 0.0444649, -0.0635343, 0.1091957
1: -0.0749285, 0.1370497, -0.0457317, 0.0752505, -0.1501790, 0.1827814
2: -0.0416235, 0.1398997, -0.0161715, 0.0679093, -0.1095328, 0.1560712
3: -0.0797155, 0.1714330, -0.0558398, 0.0794108, -0.1591263, 0.2272728
4: -0.0581345, 0.1992887, -0.0350063, 0.0803186, -0.1384531, 0.2342951

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
time: 0.28 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.45 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0037549, 0.0340886, -0.0378435, 0.0378435
1: -0.0356151, 0.0616529, -0.0356151, 0.0616529, -0.0972680, 0.0972680
2: -0.0105594, 0.0539910, -0.0105594, 0.0539910, -0.0645503, 0.0645503
3: -0.0435620, 0.0632024, -0.0435620, 0.0632024, -0.1067644, 0.1067644
4: -0.0277857, 0.0573916, -0.0277857, 0.0573916, -0.0851773, 0.0851773

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.29 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.28 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0190694, 0.1021053, -0.1058602, 0.0531580
1: -0.0356151, 0.0616529, -0.0749285, 0.1370497, -0.1726648, 0.1365814
2: -0.0105594, 0.0539910, -0.0416235, 0.1398997, -0.1504591, 0.0956145
3: -0.0435620, 0.0632024, -0.0797155, 0.1714330, -0.2149950, 0.1429179
4: -0.0277857, 0.0573916, -0.0581345, 0.1992887, -0.2270745, 0.1155261

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.29 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.27 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0037549, 0.0340886, -0.0531580, 0.1058602
1: -0.0749285, 0.1370497, -0.0356151, 0.0616529, -0.1365814, 0.1726648
2: -0.0416235, 0.1398997, -0.0105594, 0.0539910, -0.0956145, 0.1504591
3: -0.0797155, 0.1714330, -0.0435620, 0.0632024, -0.1429179, 0.2149950
4: -0.0581345, 0.1992887, -0.0277857, 0.0573916, -0.1155261, 0.2270745

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
time: 0.28 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0190694, 0.1021053, -0.1211746, 0.1211746
1: -0.0749285, 0.1370497, -0.0749285, 0.1370497, -0.2119782, 0.2119782
2: -0.0416235, 0.1398997, -0.0416235, 0.1398997, -0.1815232, 0.1815232
3: -0.0797155, 0.1714330, -0.0797155, 0.1714330, -0.2511485, 0.2511485
4: -0.0581345, 0.1992887, -0.0581345, 0.1992887, -0.2574233, 0.2574233

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
time: 0.31 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.29 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.92 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.92
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.92
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.92
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.92
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.92
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0037549, 0.0340886, -0.0376604, 0.0373968
1: -0.0350209, 0.0608409, -0.0356151, 0.0616529, -0.0966738, 0.0964560
2: -0.0102656, 0.0532160, -0.0105594, 0.0539910, -0.0642565, 0.0637753
3: -0.0427086, 0.0621664, -0.0435620, 0.0632024, -0.1059110, 0.1057284
4: -0.0273492, 0.0564452, -0.0277857, 0.0573916, -0.0847408, 0.0842309

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447883, upper bound: 0.0465151
time: 0.27 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
time: 0.29 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
time: 0.30 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0190694, 0.1021053, -0.1056770, 0.0527113
1: -0.0350209, 0.0608409, -0.0749285, 0.1370497, -0.1720706, 0.1357694
2: -0.0102656, 0.0532160, -0.0416235, 0.1398997, -0.1501652, 0.0948395
3: -0.0427086, 0.0621664, -0.0797155, 0.1714330, -0.2141416, 0.1418818
4: -0.0273492, 0.0564452, -0.0581345, 0.1992887, -0.2266379, 0.1145797

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447348, upper bound: 0.0465151
time: 0.29 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
time: 0.29 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
time: 0.27 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0037549, 0.0340886, -0.0527285, 0.1038970
1: -0.0738246, 0.1346942, -0.0356151, 0.0616529, -0.1354775, 0.1703093
2: -0.0407216, 0.1377040, -0.0105594, 0.0539910, -0.0947125, 0.1482634
3: -0.0786182, 0.1676345, -0.0435620, 0.0632024, -0.1418206, 0.2111965
4: -0.0572622, 0.1955275, -0.0277857, 0.0573916, -0.1146538, 0.2233132

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
time: 0.29 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
time: 0.29 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0190694, 0.1021053, -0.1207452, 0.1192114
1: -0.0738246, 0.1346942, -0.0749285, 0.1370497, -0.2108742, 0.2096227
2: -0.0407216, 0.1377040, -0.0416235, 0.1398997, -0.1806213, 0.1793275
3: -0.0786182, 0.1676345, -0.0797155, 0.1714330, -0.2500512, 0.2473500
4: -0.0572622, 0.1955275, -0.0581345, 0.1992887, -0.2565510, 0.2536620

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
time: 0.27 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.28 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.57 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.57
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
Binary search (step 12): status=Status.VERIFIED, low=0.1817960, high=0.1818182, mid=0.1817960, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 13) starts
Candidate diff: 0.1818071


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
time: 0.28 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.27 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.68 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0070904, 0.0444649, -0.0482198, 0.0411790
1: -0.0356151, 0.0616529, -0.0457317, 0.0752505, -0.1108656, 0.1073847
2: -0.0105594, 0.0539910, -0.0161715, 0.0679093, -0.0784687, 0.0701625
3: -0.0435620, 0.0632024, -0.0558398, 0.0794108, -0.1229728, 0.1190422
4: -0.0277857, 0.0573916, -0.0350063, 0.0803186, -0.1081043, 0.0923979

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
time: 0.27 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.26 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0070904, 0.0444649, -0.0635343, 0.1091957
1: -0.0749285, 0.1370497, -0.0457317, 0.0752505, -0.1501790, 0.1827814
2: -0.0416235, 0.1398997, -0.0161715, 0.0679093, -0.1095328, 0.1560712
3: -0.0797155, 0.1714330, -0.0558398, 0.0794108, -0.1591263, 0.2272728
4: -0.0581345, 0.1992887, -0.0350063, 0.0803186, -0.1384531, 0.2342951

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
time: 0.27 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.28 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.16 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.16
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0037549, 0.0340886, -0.0378435, 0.0378435
1: -0.0356151, 0.0616529, -0.0356151, 0.0616529, -0.0972680, 0.0972680
2: -0.0105594, 0.0539910, -0.0105594, 0.0539910, -0.0645503, 0.0645503
3: -0.0435620, 0.0632024, -0.0435620, 0.0632024, -0.1067644, 0.1067644
4: -0.0277857, 0.0573916, -0.0277857, 0.0573916, -0.0851773, 0.0851773

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.27 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.27 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0190694, 0.1021053, -0.1058602, 0.0531580
1: -0.0356151, 0.0616529, -0.0749285, 0.1370497, -0.1726648, 0.1365814
2: -0.0105594, 0.0539910, -0.0416235, 0.1398997, -0.1504591, 0.0956145
3: -0.0435620, 0.0632024, -0.0797155, 0.1714330, -0.2149950, 0.1429179
4: -0.0277857, 0.0573916, -0.0581345, 0.1992887, -0.2270745, 0.1155261

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.28 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.26 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0037549, 0.0340886, -0.0531580, 0.1058602
1: -0.0749285, 0.1370497, -0.0356151, 0.0616529, -0.1365814, 0.1726648
2: -0.0416235, 0.1398997, -0.0105594, 0.0539910, -0.0956145, 0.1504591
3: -0.0797155, 0.1714330, -0.0435620, 0.0632024, -0.1429179, 0.2149950
4: -0.0581345, 0.1992887, -0.0277857, 0.0573916, -0.1155261, 0.2270745

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
time: 0.29 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0190694, 0.1021053, -0.1211746, 0.1211746
1: -0.0749285, 0.1370497, -0.0749285, 0.1370497, -0.2119782, 0.2119782
2: -0.0416235, 0.1398997, -0.0416235, 0.1398997, -0.1815232, 0.1815232
3: -0.0797155, 0.1714330, -0.0797155, 0.1714330, -0.2511485, 0.2511485
4: -0.0581345, 0.1992887, -0.0581345, 0.1992887, -0.2574233, 0.2574233

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
time: 0.28 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.27 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.49 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0037549, 0.0340886, -0.0376604, 0.0373968
1: -0.0350209, 0.0608409, -0.0356151, 0.0616529, -0.0966738, 0.0964560
2: -0.0102656, 0.0532160, -0.0105594, 0.0539910, -0.0642565, 0.0637753
3: -0.0427086, 0.0621664, -0.0435620, 0.0632024, -0.1059110, 0.1057284
4: -0.0273492, 0.0564452, -0.0277857, 0.0573916, -0.0847408, 0.0842309

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447883, upper bound: 0.0465151
time: 0.26 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
time: 0.28 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
time: 0.29 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0190694, 0.1021053, -0.1056770, 0.0527113
1: -0.0350209, 0.0608409, -0.0749285, 0.1370497, -0.1720706, 0.1357694
2: -0.0102656, 0.0532160, -0.0416235, 0.1398997, -0.1501652, 0.0948395
3: -0.0427086, 0.0621664, -0.0797155, 0.1714330, -0.2141416, 0.1418818
4: -0.0273492, 0.0564452, -0.0581345, 0.1992887, -0.2266379, 0.1145797

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447348, upper bound: 0.0465151
time: 0.27 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
time: 0.28 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
time: 0.30 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0037549, 0.0340886, -0.0527285, 0.1038970
1: -0.0738246, 0.1346942, -0.0356151, 0.0616529, -0.1354775, 0.1703093
2: -0.0407216, 0.1377040, -0.0105594, 0.0539910, -0.0947125, 0.1482634
3: -0.0786182, 0.1676345, -0.0435620, 0.0632024, -0.1418206, 0.2111965
4: -0.0572622, 0.1955275, -0.0277857, 0.0573916, -0.1146538, 0.2233132

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
time: 0.28 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
time: 0.27 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0190694, 0.1021053, -0.1207452, 0.1192114
1: -0.0738246, 0.1346942, -0.0749285, 0.1370497, -0.2108742, 0.2096227
2: -0.0407216, 0.1377040, -0.0416235, 0.1398997, -0.1806213, 0.1793275
3: -0.0786182, 0.1676345, -0.0797155, 0.1714330, -0.2500512, 0.2473500
4: -0.0572622, 0.1955275, -0.0581345, 0.1992887, -0.2565510, 0.2536620

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
time: 0.27 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.29 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.54 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
Binary search (step 13): status=Status.VERIFIED, low=0.1818071, high=0.1818182, mid=0.1818071, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 14) starts
Candidate diff: 0.1818126


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
time: 0.29 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.27 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.72 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.72
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0070904, 0.0444649, -0.0482198, 0.0411790
1: -0.0356151, 0.0616529, -0.0457317, 0.0752505, -0.1108656, 0.1073847
2: -0.0105594, 0.0539910, -0.0161715, 0.0679093, -0.0784687, 0.0701625
3: -0.0435620, 0.0632024, -0.0558398, 0.0794108, -0.1229728, 0.1190422
4: -0.0277857, 0.0573916, -0.0350063, 0.0803186, -0.1081043, 0.0923979

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
time: 0.28 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.28 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0070904, 0.0444649, -0.0635343, 0.1091957
1: -0.0749285, 0.1370497, -0.0457317, 0.0752505, -0.1501790, 0.1827814
2: -0.0416235, 0.1398997, -0.0161715, 0.0679093, -0.1095328, 0.1560712
3: -0.0797155, 0.1714330, -0.0558398, 0.0794108, -0.1591263, 0.2272728
4: -0.0581345, 0.1992887, -0.0350063, 0.0803186, -0.1384531, 0.2342951

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
time: 0.28 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.45 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0037549, 0.0340886, -0.0378435, 0.0378435
1: -0.0356151, 0.0616529, -0.0356151, 0.0616529, -0.0972680, 0.0972680
2: -0.0105594, 0.0539910, -0.0105594, 0.0539910, -0.0645503, 0.0645503
3: -0.0435620, 0.0632024, -0.0435620, 0.0632024, -0.1067644, 0.1067644
4: -0.0277857, 0.0573916, -0.0277857, 0.0573916, -0.0851773, 0.0851773

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.28 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.28 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0190694, 0.1021053, -0.1058602, 0.0531580
1: -0.0356151, 0.0616529, -0.0749285, 0.1370497, -0.1726648, 0.1365814
2: -0.0105594, 0.0539910, -0.0416235, 0.1398997, -0.1504591, 0.0956145
3: -0.0435620, 0.0632024, -0.0797155, 0.1714330, -0.2149950, 0.1429179
4: -0.0277857, 0.0573916, -0.0581345, 0.1992887, -0.2270745, 0.1155261

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.30 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.27 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0037549, 0.0340886, -0.0531580, 0.1058602
1: -0.0749285, 0.1370497, -0.0356151, 0.0616529, -0.1365814, 0.1726648
2: -0.0416235, 0.1398997, -0.0105594, 0.0539910, -0.0956145, 0.1504591
3: -0.0797155, 0.1714330, -0.0435620, 0.0632024, -0.1429179, 0.2149950
4: -0.0581345, 0.1992887, -0.0277857, 0.0573916, -0.1155261, 0.2270745

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
time: 0.30 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0190694, 0.1021053, -0.1211746, 0.1211746
1: -0.0749285, 0.1370497, -0.0749285, 0.1370497, -0.2119782, 0.2119782
2: -0.0416235, 0.1398997, -0.0416235, 0.1398997, -0.1815232, 0.1815232
3: -0.0797155, 0.1714330, -0.0797155, 0.1714330, -0.2511485, 0.2511485
4: -0.0581345, 0.1992887, -0.0581345, 0.1992887, -0.2574233, 0.2574233

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
time: 0.30 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.29 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.84 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.84
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.84
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.84
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.84
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.84
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0037549, 0.0340886, -0.0376604, 0.0373968
1: -0.0350209, 0.0608409, -0.0356151, 0.0616529, -0.0966738, 0.0964560
2: -0.0102656, 0.0532160, -0.0105594, 0.0539910, -0.0642565, 0.0637753
3: -0.0427086, 0.0621664, -0.0435620, 0.0632024, -0.1059110, 0.1057284
4: -0.0273492, 0.0564452, -0.0277857, 0.0573916, -0.0847408, 0.0842309

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447883, upper bound: 0.0465151
time: 0.27 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
time: 0.29 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
time: 0.30 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0190694, 0.1021053, -0.1056770, 0.0527113
1: -0.0350209, 0.0608409, -0.0749285, 0.1370497, -0.1720706, 0.1357694
2: -0.0102656, 0.0532160, -0.0416235, 0.1398997, -0.1501652, 0.0948395
3: -0.0427086, 0.0621664, -0.0797155, 0.1714330, -0.2141416, 0.1418818
4: -0.0273492, 0.0564452, -0.0581345, 0.1992887, -0.2266379, 0.1145797

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447348, upper bound: 0.0465151
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
time: 0.30 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
time: 0.30 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0037549, 0.0340886, -0.0527285, 0.1038970
1: -0.0738246, 0.1346942, -0.0356151, 0.0616529, -0.1354775, 0.1703093
2: -0.0407216, 0.1377040, -0.0105594, 0.0539910, -0.0947125, 0.1482634
3: -0.0786182, 0.1676345, -0.0435620, 0.0632024, -0.1418206, 0.2111965
4: -0.0572622, 0.1955275, -0.0277857, 0.0573916, -0.1146538, 0.2233132

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
time: 0.29 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
time: 0.28 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0190694, 0.1021053, -0.1207452, 0.1192114
1: -0.0738246, 0.1346942, -0.0749285, 0.1370497, -0.2108742, 0.2096227
2: -0.0407216, 0.1377040, -0.0416235, 0.1398997, -0.1806213, 0.1793275
3: -0.0786182, 0.1676345, -0.0797155, 0.1714330, -0.2500512, 0.2473500
4: -0.0572622, 0.1955275, -0.0581345, 0.1992887, -0.2565510, 0.2536620

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
time: 0.27 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.27 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.55 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
Binary search (step 14): status=Status.VERIFIED, low=0.1818126, high=0.1818182, mid=0.1818126, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 15) starts
Candidate diff: 0.1818154


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
time: 0.28 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.26 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.68 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0070904, 0.0444649, -0.0482198, 0.0411790
1: -0.0356151, 0.0616529, -0.0457317, 0.0752505, -0.1108656, 0.1073847
2: -0.0105594, 0.0539910, -0.0161715, 0.0679093, -0.0784687, 0.0701625
3: -0.0435620, 0.0632024, -0.0558398, 0.0794108, -0.1229728, 0.1190422
4: -0.0277857, 0.0573916, -0.0350063, 0.0803186, -0.1081043, 0.0923979

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
time: 0.27 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.26 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0070904, 0.0444649, -0.0635343, 0.1091957
1: -0.0749285, 0.1370497, -0.0457317, 0.0752505, -0.1501790, 0.1827814
2: -0.0416235, 0.1398997, -0.0161715, 0.0679093, -0.1095328, 0.1560712
3: -0.0797155, 0.1714330, -0.0558398, 0.0794108, -0.1591263, 0.2272728
4: -0.0581345, 0.1992887, -0.0350063, 0.0803186, -0.1384531, 0.2342951

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
time: 0.27 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.28 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.19 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0037549, 0.0340886, -0.0378435, 0.0378435
1: -0.0356151, 0.0616529, -0.0356151, 0.0616529, -0.0972680, 0.0972680
2: -0.0105594, 0.0539910, -0.0105594, 0.0539910, -0.0645503, 0.0645503
3: -0.0435620, 0.0632024, -0.0435620, 0.0632024, -0.1067644, 0.1067644
4: -0.0277857, 0.0573916, -0.0277857, 0.0573916, -0.0851773, 0.0851773

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.27 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.27 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0190694, 0.1021053, -0.1058602, 0.0531580
1: -0.0356151, 0.0616529, -0.0749285, 0.1370497, -0.1726648, 0.1365814
2: -0.0105594, 0.0539910, -0.0416235, 0.1398997, -0.1504591, 0.0956145
3: -0.0435620, 0.0632024, -0.0797155, 0.1714330, -0.2149950, 0.1429179
4: -0.0277857, 0.0573916, -0.0581345, 0.1992887, -0.2270745, 0.1155261

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.28 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.26 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0037549, 0.0340886, -0.0531580, 0.1058602
1: -0.0749285, 0.1370497, -0.0356151, 0.0616529, -0.1365814, 0.1726648
2: -0.0416235, 0.1398997, -0.0105594, 0.0539910, -0.0956145, 0.1504591
3: -0.0797155, 0.1714330, -0.0435620, 0.0632024, -0.1429179, 0.2149950
4: -0.0581345, 0.1992887, -0.0277857, 0.0573916, -0.1155261, 0.2270745

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
time: 0.27 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0190694, 0.1021053, -0.1211746, 0.1211746
1: -0.0749285, 0.1370497, -0.0749285, 0.1370497, -0.2119782, 0.2119782
2: -0.0416235, 0.1398997, -0.0416235, 0.1398997, -0.1815232, 0.1815232
3: -0.0797155, 0.1714330, -0.0797155, 0.1714330, -0.2511485, 0.2511485
4: -0.0581345, 0.1992887, -0.0581345, 0.1992887, -0.2574233, 0.2574233

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
time: 0.28 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.27 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.50 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.50
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.50
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.50
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.50
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.50
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0037549, 0.0340886, -0.0376604, 0.0373968
1: -0.0350209, 0.0608409, -0.0356151, 0.0616529, -0.0966738, 0.0964560
2: -0.0102656, 0.0532160, -0.0105594, 0.0539910, -0.0642565, 0.0637753
3: -0.0427086, 0.0621664, -0.0435620, 0.0632024, -0.1059110, 0.1057284
4: -0.0273492, 0.0564452, -0.0277857, 0.0573916, -0.0847408, 0.0842309

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447883, upper bound: 0.0465151
time: 0.26 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
time: 0.27 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
time: 0.28 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0190694, 0.1021053, -0.1056770, 0.0527113
1: -0.0350209, 0.0608409, -0.0749285, 0.1370497, -0.1720706, 0.1357694
2: -0.0102656, 0.0532160, -0.0416235, 0.1398997, -0.1501652, 0.0948395
3: -0.0427086, 0.0621664, -0.0797155, 0.1714330, -0.2141416, 0.1418818
4: -0.0273492, 0.0564452, -0.0581345, 0.1992887, -0.2266379, 0.1145797

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447348, upper bound: 0.0465151
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
time: 0.29 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
time: 0.27 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0037549, 0.0340886, -0.0527285, 0.1038970
1: -0.0738246, 0.1346942, -0.0356151, 0.0616529, -0.1354775, 0.1703093
2: -0.0407216, 0.1377040, -0.0105594, 0.0539910, -0.0947125, 0.1482634
3: -0.0786182, 0.1676345, -0.0435620, 0.0632024, -0.1418206, 0.2111965
4: -0.0572622, 0.1955275, -0.0277857, 0.0573916, -0.1146538, 0.2233132

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
time: 0.28 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
time: 0.28 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0190694, 0.1021053, -0.1207452, 0.1192114
1: -0.0738246, 0.1346942, -0.0749285, 0.1370497, -0.2108742, 0.2096227
2: -0.0407216, 0.1377040, -0.0416235, 0.1398997, -0.1806213, 0.1793275
3: -0.0786182, 0.1676345, -0.0797155, 0.1714330, -0.2500512, 0.2473500
4: -0.0572622, 0.1955275, -0.0581345, 0.1992887, -0.2565510, 0.2536620

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
time: 0.27 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.28 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.54 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
Binary search (step 15): status=Status.VERIFIED, low=0.1818154, high=0.1818182, mid=0.1818154, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 16) starts
Candidate diff: 0.1818168


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
time: 0.29 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.27 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.71 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.71
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.71
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0070904, 0.0444649, -0.0482198, 0.0411790
1: -0.0356151, 0.0616529, -0.0457317, 0.0752505, -0.1108656, 0.1073847
2: -0.0105594, 0.0539910, -0.0161715, 0.0679093, -0.0784687, 0.0701625
3: -0.0435620, 0.0632024, -0.0558398, 0.0794108, -0.1229728, 0.1190422
4: -0.0277857, 0.0573916, -0.0350063, 0.0803186, -0.1081043, 0.0923979

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
time: 0.28 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.28 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0070904, 0.0444649, -0.0635343, 0.1091957
1: -0.0749285, 0.1370497, -0.0457317, 0.0752505, -0.1501790, 0.1827814
2: -0.0416235, 0.1398997, -0.0161715, 0.0679093, -0.1095328, 0.1560712
3: -0.0797155, 0.1714330, -0.0558398, 0.0794108, -0.1591263, 0.2272728
4: -0.0581345, 0.1992887, -0.0350063, 0.0803186, -0.1384531, 0.2342951

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
time: 0.29 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.45 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.45
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0037549, 0.0340886, -0.0378435, 0.0378435
1: -0.0356151, 0.0616529, -0.0356151, 0.0616529, -0.0972680, 0.0972680
2: -0.0105594, 0.0539910, -0.0105594, 0.0539910, -0.0645503, 0.0645503
3: -0.0435620, 0.0632024, -0.0435620, 0.0632024, -0.1067644, 0.1067644
4: -0.0277857, 0.0573916, -0.0277857, 0.0573916, -0.0851773, 0.0851773

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.28 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.28 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0190694, 0.1021053, -0.1058602, 0.0531580
1: -0.0356151, 0.0616529, -0.0749285, 0.1370497, -0.1726648, 0.1365814
2: -0.0105594, 0.0539910, -0.0416235, 0.1398997, -0.1504591, 0.0956145
3: -0.0435620, 0.0632024, -0.0797155, 0.1714330, -0.2149950, 0.1429179
4: -0.0277857, 0.0573916, -0.0581345, 0.1992887, -0.2270745, 0.1155261

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.29 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.28 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0037549, 0.0340886, -0.0531580, 0.1058602
1: -0.0749285, 0.1370497, -0.0356151, 0.0616529, -0.1365814, 0.1726648
2: -0.0416235, 0.1398997, -0.0105594, 0.0539910, -0.0956145, 0.1504591
3: -0.0797155, 0.1714330, -0.0435620, 0.0632024, -0.1429179, 0.2149950
4: -0.0581345, 0.1992887, -0.0277857, 0.0573916, -0.1155261, 0.2270745

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
time: 0.28 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.34 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0190694, 0.1021053, -0.1211746, 0.1211746
1: -0.0749285, 0.1370497, -0.0749285, 0.1370497, -0.2119782, 0.2119782
2: -0.0416235, 0.1398997, -0.0416235, 0.1398997, -0.1815232, 0.1815232
3: -0.0797155, 0.1714330, -0.0797155, 0.1714330, -0.2511485, 0.2511485
4: -0.0581345, 0.1992887, -0.0581345, 0.1992887, -0.2574233, 0.2574233

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
time: 0.29 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.28 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.79 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.79
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.79
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.79
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.79
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.79
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0037549, 0.0340886, -0.0376604, 0.0373968
1: -0.0350209, 0.0608409, -0.0356151, 0.0616529, -0.0966738, 0.0964560
2: -0.0102656, 0.0532160, -0.0105594, 0.0539910, -0.0642565, 0.0637753
3: -0.0427086, 0.0621664, -0.0435620, 0.0632024, -0.1059110, 0.1057284
4: -0.0273492, 0.0564452, -0.0277857, 0.0573916, -0.0847408, 0.0842309

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447883, upper bound: 0.0465151
time: 0.27 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
time: 0.28 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
time: 0.30 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0190694, 0.1021053, -0.1056770, 0.0527113
1: -0.0350209, 0.0608409, -0.0749285, 0.1370497, -0.1720706, 0.1357694
2: -0.0102656, 0.0532160, -0.0416235, 0.1398997, -0.1501652, 0.0948395
3: -0.0427086, 0.0621664, -0.0797155, 0.1714330, -0.2141416, 0.1418818
4: -0.0273492, 0.0564452, -0.0581345, 0.1992887, -0.2266379, 0.1145797

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447348, upper bound: 0.0465151
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
time: 0.30 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
time: 0.29 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0037549, 0.0340886, -0.0527285, 0.1038970
1: -0.0738246, 0.1346942, -0.0356151, 0.0616529, -0.1354775, 0.1703093
2: -0.0407216, 0.1377040, -0.0105594, 0.0539910, -0.0947125, 0.1482634
3: -0.0786182, 0.1676345, -0.0435620, 0.0632024, -0.1418206, 0.2111965
4: -0.0572622, 0.1955275, -0.0277857, 0.0573916, -0.1146538, 0.2233132

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
time: 0.29 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
time: 0.28 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0190694, 0.1021053, -0.1207452, 0.1192114
1: -0.0738246, 0.1346942, -0.0749285, 0.1370497, -0.2108742, 0.2096227
2: -0.0407216, 0.1377040, -0.0416235, 0.1398997, -0.1806213, 0.1793275
3: -0.0786182, 0.1676345, -0.0797155, 0.1714330, -0.2500512, 0.2473500
4: -0.0572622, 0.1955275, -0.0581345, 0.1992887, -0.2565510, 0.2536620

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
time: 0.27 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.28 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.54 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
Binary search (step 16): status=Status.VERIFIED, low=0.1818168, high=0.1818182, mid=0.1818168, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary search (step 17) starts
Candidate diff: 0.1818175


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
time: 0.27 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.25 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.66 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.66
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0465606
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.66
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0070904, 0.0444649, -0.0482198, 0.0411790
1: -0.0356151, 0.0616529, -0.0457317, 0.0752505, -0.1108656, 0.1073847
2: -0.0105594, 0.0539910, -0.0161715, 0.0679093, -0.0784687, 0.0701625
3: -0.0435620, 0.0632024, -0.0558398, 0.0794108, -0.1229728, 0.1190422
4: -0.0277857, 0.0573916, -0.0350063, 0.0803186, -0.1081043, 0.0923979

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
time: 0.27 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
time: 0.26 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0070904, 0.0444649, -0.0635343, 0.1091957
1: -0.0749285, 0.1370497, -0.0457317, 0.0752505, -0.1501790, 0.1827814
2: -0.0416235, 0.1398997, -0.0161715, 0.0679093, -0.1095328, 0.1560712
3: -0.0797155, 0.1714330, -0.0558398, 0.0794108, -0.1591263, 0.2272728
4: -0.0581345, 0.1992887, -0.0350063, 0.0803186, -0.1384531, 0.2342951

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
time: 0.27 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918
time: 0.28 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.14 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462906
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -0.0462906, upper bound: 0.0462918
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462906
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.14
Output dim: 0, lower bound: -0.0462918, upper bound: 0.0462918

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0037549, 0.0340886, -0.0378435, 0.0378435
1: -0.0356151, 0.0616529, -0.0356151, 0.0616529, -0.0972680, 0.0972680
2: -0.0105594, 0.0539910, -0.0105594, 0.0539910, -0.0645503, 0.0645503
3: -0.0435620, 0.0632024, -0.0435620, 0.0632024, -0.1067644, 0.1067644
4: -0.0277857, 0.0573916, -0.0277857, 0.0573916, -0.0851773, 0.0851773

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.26 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.26 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0037549, 0.0340886, -0.0190694, 0.1021053, -0.1058602, 0.0531580
1: -0.0356151, 0.0616529, -0.0749285, 0.1370497, -0.1726648, 0.1365814
2: -0.0105594, 0.0539910, -0.0416235, 0.1398997, -0.1504591, 0.0956145
3: -0.0435620, 0.0632024, -0.0797155, 0.1714330, -0.2149950, 0.1429179
4: -0.0277857, 0.0573916, -0.0581345, 0.1992887, -0.2270745, 0.1155261

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
time: 0.28 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.26 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0037549, 0.0340886, -0.0531580, 0.1058602
1: -0.0749285, 0.1370497, -0.0356151, 0.0616529, -0.1365814, 0.1726648
2: -0.0416235, 0.1398997, -0.0105594, 0.0539910, -0.0956145, 0.1504591
3: -0.0797155, 0.1714330, -0.0435620, 0.0632024, -0.1429179, 0.2149950
4: -0.0581345, 0.1992887, -0.0277857, 0.0573916, -0.1155261, 0.2270745

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
time: 0.28 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.32 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0190694, 0.1021053, -0.0190694, 0.1021053, -0.1211746, 0.1211746
1: -0.0749285, 0.1370497, -0.0749285, 0.1370497, -0.2119782, 0.2119782
2: -0.0416235, 0.1398997, -0.0416235, 0.1398997, -0.1815232, 0.1815232
3: -0.0797155, 0.1714330, -0.0797155, 0.1714330, -0.2511485, 0.2511485
4: -0.0581345, 0.1992887, -0.0581345, 0.1992887, -0.2574233, 0.2574233

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
time: 0.29 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.27 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.49 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0465221
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0462533
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.49
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0037549, 0.0340886, -0.0376604, 0.0373968
1: -0.0350209, 0.0608409, -0.0356151, 0.0616529, -0.0966738, 0.0964560
2: -0.0102656, 0.0532160, -0.0105594, 0.0539910, -0.0642565, 0.0637753
3: -0.0427086, 0.0621664, -0.0435620, 0.0632024, -0.1059110, 0.1057284
4: -0.0273492, 0.0564452, -0.0277857, 0.0573916, -0.0847408, 0.0842309

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447883, upper bound: 0.0465151
time: 0.26 seconds

## Relational analysis of IS_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
time: 0.27 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
time: 0.28 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0035717, 0.0336419, -0.0190694, 0.1021053, -0.1056770, 0.0527113
1: -0.0350209, 0.0608409, -0.0749285, 0.1370497, -0.1720706, 0.1357694
2: -0.0102656, 0.0532160, -0.0416235, 0.1398997, -0.1501652, 0.0948395
3: -0.0427086, 0.0621664, -0.0797155, 0.1714330, -0.2141416, 0.1418818
4: -0.0273492, 0.0564452, -0.0581345, 0.1992887, -0.2266379, 0.1145797

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0447348, upper bound: 0.0465151
time: 0.27 seconds

## Relational analysis of IS_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
time: 0.29 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
time: 0.27 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0037549, 0.0340886, -0.0527285, 0.1038970
1: -0.0738246, 0.1346942, -0.0356151, 0.0616529, -0.1354775, 0.1703093
2: -0.0407216, 0.1377040, -0.0105594, 0.0539910, -0.0947125, 0.1482634
3: -0.0786182, 0.1676345, -0.0435620, 0.0632024, -0.1418206, 0.2111965
4: -0.0572622, 0.1955275, -0.0277857, 0.0573916, -0.1146538, 0.2233132

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
time: 0.28 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
time: 0.28 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0186399, 0.1001420, -0.0190694, 0.1021053, -0.1207452, 0.1192114
1: -0.0738246, 0.1346942, -0.0749285, 0.1370497, -0.2108742, 0.2096227
2: -0.0407216, 0.1377040, -0.0416235, 0.1398997, -0.1806213, 0.1793275
3: -0.0786182, 0.1676345, -0.0797155, 0.1714330, -0.2500512, 0.2473500
4: -0.0572622, 0.1955275, -0.0581345, 0.1992887, -0.2565510, 0.2536620

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
time: 0.27 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.28 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.51 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.51
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
Binary search (step 17): status=Status.VERIFIED, low=0.1818175, high=0.1818182, mid=0.1818175, abs_max=0.05155529826879501
rel_dist={0: [-0.04657255964042008, 0.04657255964042008]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.1818174936554442
execution time: 544.90 seconds
