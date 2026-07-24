## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.045175422


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0070904, 0.0444649, -0.0070904, 0.0444649, -0.0515553, 0.0515553)
1: (-0.0457317, 0.0752505, -0.0457317, 0.0752505, -0.1209822, 0.1209822)
2: (-0.0161715, 0.0679093, -0.0161715, 0.0679093, -0.0840808, 0.0840808)
3: (-0.0558398, 0.0794108, -0.0558398, 0.0794108, -0.1352506, 0.1352506)
4: (-0.0350063, 0.0803186, -0.0350063, 0.0803186, -0.1153249, 0.1153249)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.70 + 0.76 = 1.46 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0465726, upper bound: 0.0465726

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462715, upper bound: 0.0465606
time: 0.23 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462715, upper bound: 0.0462716
time: 0.22 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.51 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.51
Output dim: 0, lower bound: -0.0462715, upper bound: 0.0465606
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.51
Output dim: 0, lower bound: -0.0462715, upper bound: 0.0462716

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0037549, 0.0340886, -0.0070904, 0.0444649, -0.0482198, 0.0411790
1: -0.0356151, 0.0616529, -0.0457317, 0.0752505, -0.1108656, 0.1073847
2: -0.0105594, 0.0539910, -0.0161715, 0.0679093, -0.0784687, 0.0701625
3: -0.0435620, 0.0632024, -0.0558398, 0.0794108, -0.1229728, 0.1190422
4: -0.0277857, 0.0573916, -0.0350063, 0.0803186, -0.1081043, 0.0923979

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462715, upper bound: 0.0462715
time: 0.20 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462715, upper bound: 0.0462716
time: 0.18 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0190694, 0.1021053, -0.0070904, 0.0444649, -0.0635343, 0.1091957
1: -0.0749285, 0.1370497, -0.0457317, 0.0752505, -0.1501790, 0.1827814
2: -0.0416235, 0.1398997, -0.0161715, 0.0679093, -0.1095328, 0.1560712
3: -0.0797155, 0.1714330, -0.0558398, 0.0794108, -0.1591263, 0.2272728
4: -0.0581345, 0.1992887, -0.0350063, 0.0803186, -0.1384531, 0.2342951

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462716, upper bound: 0.0462715
time: 0.20 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0462716, upper bound: 0.0462716
time: 0.19 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.11 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1.11
Output dim: 0, lower bound: -0.0462715, upper bound: 0.0462715
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.11
Output dim: 0, lower bound: -0.0462715, upper bound: 0.0462716
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1.11
Output dim: 0, lower bound: -0.0462716, upper bound: 0.0462715
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1.11
Output dim: 0, lower bound: -0.0462716, upper bound: 0.0462716

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0037549, 0.0340886, -0.0037549, 0.0340886, -0.0378435, 0.0378435
1: -0.0356151, 0.0616529, -0.0356151, 0.0616529, -0.0972680, 0.0972680
2: -0.0105594, 0.0539910, -0.0105594, 0.0539910, -0.0645503, 0.0645503
3: -0.0435620, 0.0632024, -0.0435620, 0.0632024, -0.1067644, 0.1067644
4: -0.0277857, 0.0573916, -0.0277857, 0.0573916, -0.0851773, 0.0851773

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0464564
time: 0.18 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.19 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0037549, 0.0340886, -0.0190694, 0.1021053, -0.1058602, 0.0531580
1: -0.0356151, 0.0616529, -0.0749285, 0.1370497, -0.1726648, 0.1365814
2: -0.0105594, 0.0539910, -0.0416235, 0.1398997, -0.1504591, 0.0956145
3: -0.0435620, 0.0632024, -0.0797155, 0.1714330, -0.2149950, 0.1429179
4: -0.0277857, 0.0573916, -0.0581345, 0.1992887, -0.2270745, 0.1155261

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0464566
time: 0.20 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
time: 0.18 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0190694, 0.1021053, -0.0037549, 0.0340886, -0.0531580, 0.1058602
1: -0.0749285, 0.1370497, -0.0356151, 0.0616529, -0.1365814, 0.1726648
2: -0.0416235, 0.1398997, -0.0105594, 0.0539910, -0.0956145, 0.1504591
3: -0.0797155, 0.1714330, -0.0435620, 0.0632024, -0.1429179, 0.2149950
4: -0.0581345, 0.1992887, -0.0277857, 0.0573916, -0.1155261, 0.2270745

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0460547
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.24 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0190694, 0.1021053, -0.0190694, 0.1021053, -0.1211746, 0.1211746
1: -0.0749285, 0.1370497, -0.0749285, 0.1370497, -0.2119782, 0.2119782
2: -0.0416235, 0.1398997, -0.0416235, 0.1398997, -0.1815232, 0.1815232
3: -0.0797155, 0.1714330, -0.0797155, 0.1714330, -0.2511485, 0.2511485
4: -0.0581345, 0.1992887, -0.0581345, 0.1992887, -0.2574233, 0.2574233

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0460547
time: 0.19 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
time: 0.19 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.31 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.31
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0464564
NS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 1.31
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.31
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0464566
NS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 1.31
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0450072
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.31
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0460547
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 1.31
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.31
Output dim: 0, lower bound: -0.0449615, upper bound: 0.0460547
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 1.31
Output dim: 0, lower bound: -0.0449498, upper bound: 0.0449499

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0035717, 0.0336419, -0.0037549, 0.0340886, -0.0376604, 0.0373968
1: -0.0350209, 0.0608409, -0.0356151, 0.0616529, -0.0966738, 0.0964560
2: -0.0102656, 0.0532160, -0.0105594, 0.0539910, -0.0642565, 0.0637753
3: -0.0427086, 0.0621664, -0.0435620, 0.0632024, -0.1059110, 0.1057284
4: -0.0273492, 0.0564452, -0.0277857, 0.0573916, -0.0847408, 0.0842309

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
time: 0.20 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
time: 0.18 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0035717, 0.0336419, -0.0190694, 0.1021053, -0.1056770, 0.0527113
1: -0.0350209, 0.0608409, -0.0749285, 0.1370497, -0.1720706, 0.1357694
2: -0.0102656, 0.0532160, -0.0416235, 0.1398997, -0.1501652, 0.0948395
3: -0.0427086, 0.0621664, -0.0797155, 0.1714330, -0.2141416, 0.1418818
4: -0.0273492, 0.0564452, -0.0581345, 0.1992887, -0.2266379, 0.1145797

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
time: 0.21 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
time: 0.19 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0186399, 0.1001420, -0.0037549, 0.0340886, -0.0527285, 0.1038970
1: -0.0738246, 0.1346942, -0.0356151, 0.0616529, -0.1354775, 0.1703093
2: -0.0407216, 0.1377040, -0.0105594, 0.0539910, -0.0947125, 0.1482634
3: -0.0786182, 0.1676345, -0.0435620, 0.0632024, -0.1418206, 0.2111965
4: -0.0572622, 0.1955275, -0.0277857, 0.0573916, -0.1146538, 0.2233132

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
time: 0.20 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0186399, 0.1001420, -0.0190694, 0.1021053, -0.1207452, 0.1192114
1: -0.0738246, 0.1346942, -0.0749285, 0.1370497, -0.2108742, 0.2096227
2: -0.0407216, 0.1377040, -0.0416235, 0.1398997, -0.1806213, 0.1793275
3: -0.0786182, 0.1676345, -0.0797155, 0.1714330, -0.2500512, 0.2473500
4: -0.0572622, 0.1955275, -0.0581345, 0.1992887, -0.2565510, 0.2536620

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
time: 0.19 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499
time: 0.19 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.12 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 1.12
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449894
NS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 1.12
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0450072
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 1.12
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449894
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 1.12
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0450072
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 1.12
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449321
NS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 1.12
Output dim: 0, lower bound: -0.0449894, upper bound: 0.0449499
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 1.12
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449321
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 1.12
Output dim: 0, lower bound: -0.0449321, upper bound: 0.0449499

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 1.46 + 12.47 = 13.93 seconds
