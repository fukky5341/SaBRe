## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.07289891999999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0313130, 0.1384850, 0.0313130, 0.1384850, -0.1071720, 0.1071720)
1: (-0.0147149, 0.0132303, -0.0147149, 0.0132303, -0.0279452, 0.0279452)
2: (0.0269715, 0.0536092, 0.0269715, 0.0536092, -0.0266376, 0.0266376)
3: (-0.0199371, -0.0015285, -0.0199371, -0.0015285, -0.0184086, 0.0184086)
4: (0.0235005, 0.0482717, 0.0235005, 0.0482717, -0.0247711, 0.0247711)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.84 + 0.66 = 2.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0809988, upper bound: 0.0809988

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0804479, upper bound: 0.0807474
time: 0.22 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0809970, upper bound: 0.0809970
time: 0.23 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.61 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0804479, upper bound: 0.0807474
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.61
Output dim: 0, lower bound: -0.0809970, upper bound: 0.0809970

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.0370888, 0.0778702, 0.0313130, 0.1384850, -0.1013962, 0.0465572
1: -0.0059404, 0.0113792, -0.0147149, 0.0132303, -0.0191707, 0.0260941
2: 0.0328795, 0.0515916, 0.0269715, 0.0536092, -0.0207297, 0.0246201
3: -0.0178043, -0.0074071, -0.0199371, -0.0015285, -0.0162758, 0.0125300
4: 0.0277665, 0.0460963, 0.0235005, 0.0482717, -0.0205051, 0.0225958

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0803138, upper bound: 0.0803138
time: 0.23 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0803138, upper bound: 0.0803138
time: 0.24 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.0319700, 0.1270105, 0.0313130, 0.1384850, -0.1065149, 0.0956975
1: -0.0133929, 0.0117421, -0.0147149, 0.0132303, -0.0266232, 0.0264570
2: 0.0280271, 0.0521042, 0.0269715, 0.0536092, -0.0255821, 0.0251327
3: -0.0194361, -0.0026731, -0.0199371, -0.0015285, -0.0179076, 0.0172640
4: 0.0244007, 0.0470015, 0.0235005, 0.0482717, -0.0238709, 0.0235010

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0807474, upper bound: 0.0804479
time: 0.22 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0807474, upper bound: 0.0804479
time: 0.22 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.61 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 0, lower bound: -0.0803138, upper bound: 0.0803138
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 0, lower bound: -0.0803138, upper bound: 0.0803138
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 0, lower bound: -0.0807474, upper bound: 0.0804479
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.61
Output dim: 0, lower bound: -0.0807474, upper bound: 0.0804479

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.0370888, 0.0778702, 0.0370888, 0.0778702, -0.0407814, 0.0407814
1: -0.0059404, 0.0113792, -0.0059404, 0.0113792, -0.0173196, 0.0173196
2: 0.0328795, 0.0515916, 0.0328795, 0.0515916, -0.0187122, 0.0187122
3: -0.0178043, -0.0074071, -0.0178043, -0.0074071, -0.0103972, 0.0103972
4: 0.0277665, 0.0460963, 0.0277665, 0.0460963, -0.0183298, 0.0183298

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0767145, upper bound: 0.0802982
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0767639, upper bound: 0.0767639
time: 0.22 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 0.0370888, 0.0778702, 0.0319700, 0.1270105, -0.0899217, 0.0459002
1: -0.0059404, 0.0113792, -0.0133929, 0.0117421, -0.0176825, 0.0247721
2: 0.0328795, 0.0515916, 0.0280271, 0.0521042, -0.0192247, 0.0235645
3: -0.0178043, -0.0074071, -0.0194361, -0.0026731, -0.0151312, 0.0120290
4: 0.0277665, 0.0460963, 0.0244007, 0.0470015, -0.0192350, 0.0216956

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0784942, upper bound: 0.0799558
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0778046, upper bound: 0.0783930
time: 0.21 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.0319700, 0.1270105, 0.0370888, 0.0778702, -0.0459002, 0.0899217
1: -0.0133929, 0.0117421, -0.0059404, 0.0113792, -0.0247721, 0.0176825
2: 0.0280271, 0.0521042, 0.0328795, 0.0515916, -0.0235645, 0.0192247
3: -0.0194361, -0.0026731, -0.0178043, -0.0074071, -0.0120290, 0.0151312
4: 0.0244007, 0.0470015, 0.0277665, 0.0460963, -0.0216956, 0.0192350

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0799558, upper bound: 0.0803098
time: 0.24 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0783930, upper bound: 0.0794123
time: 0.22 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.0319700, 0.1270105, 0.0319700, 0.1270105, -0.0950404, 0.0950404
1: -0.0133929, 0.0117421, -0.0133929, 0.0117421, -0.0251350, 0.0251350
2: 0.0280271, 0.0521042, 0.0280271, 0.0521042, -0.0240771, 0.0240771
3: -0.0194361, -0.0026731, -0.0194361, -0.0026731, -0.0167630, 0.0167630
4: 0.0244007, 0.0470015, 0.0244007, 0.0470015, -0.0226008, 0.0226008

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0769182, upper bound: 0.0806794
time: 0.24 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0770970, upper bound: 0.0778447
time: 0.23 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.30 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -0.0767145, upper bound: 0.0802982
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -0.0767639, upper bound: 0.0767639
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -0.0784942, upper bound: 0.0799558
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -0.0778046, upper bound: 0.0783930
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -0.0799558, upper bound: 0.0803098
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -0.0783930, upper bound: 0.0794123
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -0.0769182, upper bound: 0.0806794
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -0.0770970, upper bound: 0.0778447

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0373425, 0.0757578, 0.0370888, 0.0778702, -0.0405278, 0.0386690
1: -0.0053885, 0.0112061, -0.0059404, 0.0113792, -0.0167677, 0.0171465
2: 0.0331099, 0.0513776, 0.0328795, 0.0515916, -0.0184817, 0.0184981
3: -0.0175277, -0.0075475, -0.0178043, -0.0074071, -0.0101206, 0.0102568
4: 0.0279520, 0.0458557, 0.0277665, 0.0460963, -0.0181443, 0.0180892

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0767145, upper bound: 0.0767145
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0767145, upper bound: 0.0767639
time: 0.26 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0361534, 0.1192177, 0.0370888, 0.0778702, -0.0417169, 0.0821289
1: -0.0082976, 0.0143075, -0.0059404, 0.0113792, -0.0196768, 0.0202479
2: 0.0316524, 0.0546903, 0.0328795, 0.0515916, -0.0199393, 0.0218108
3: -0.0182268, -0.0039084, -0.0178043, -0.0074071, -0.0108198, 0.0138959
4: 0.0268038, 0.0493851, 0.0277665, 0.0460963, -0.0192925, 0.0216186

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0767639, upper bound: 0.0767145
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0767639, upper bound: 0.0767639
time: 0.23 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0372057, 0.0714706, 0.0319700, 0.1270105, -0.0898048, 0.0395005
1: -0.0056048, 0.0110969, -0.0133929, 0.0117421, -0.0173470, 0.0244898
2: 0.0330311, 0.0513681, 0.0280271, 0.0521042, -0.0190732, 0.0233410
3: -0.0177317, -0.0077312, -0.0194361, -0.0026731, -0.0150586, 0.0117049
4: 0.0279229, 0.0458125, 0.0244007, 0.0470015, -0.0190786, 0.0214118

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0802910, upper bound: 0.0799221
time: 0.23 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0803042, upper bound: 0.0798344
time: 0.23 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0371996, 0.0817597, 0.0319700, 0.1270105, -0.0898109, 0.0497896
1: -0.0060085, 0.0113276, -0.0133929, 0.0117421, -0.0177507, 0.0247205
2: 0.0329229, 0.0514867, 0.0280271, 0.0521042, -0.0191813, 0.0234596
3: -0.0178054, -0.0073031, -0.0194361, -0.0026731, -0.0151323, 0.0121330
4: 0.0277720, 0.0460424, 0.0244007, 0.0470015, -0.0192295, 0.0216417

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0794123, upper bound: 0.0783930
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0794123, upper bound: 0.0783930
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0319700, 0.1270105, 0.0372057, 0.0714706, -0.0395005, 0.0898048
1: -0.0133929, 0.0117421, -0.0056048, 0.0110969, -0.0244898, 0.0173470
2: 0.0280271, 0.0521042, 0.0330311, 0.0513681, -0.0233410, 0.0190732
3: -0.0194361, -0.0026731, -0.0177317, -0.0077312, -0.0117049, 0.0150586
4: 0.0244007, 0.0470015, 0.0279229, 0.0458125, -0.0214118, 0.0190786

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0799221, upper bound: 0.0802910
time: 0.24 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0798344, upper bound: 0.0803042
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.0319700, 0.1270105, 0.0371996, 0.0817597, -0.0497896, 0.0898109
1: -0.0133929, 0.0117421, -0.0060085, 0.0113276, -0.0247205, 0.0177507
2: 0.0280271, 0.0521042, 0.0329229, 0.0514867, -0.0234596, 0.0191813
3: -0.0194361, -0.0026731, -0.0178054, -0.0073031, -0.0121330, 0.0151323
4: 0.0244007, 0.0470015, 0.0277720, 0.0460424, -0.0216417, 0.0192295

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0783930, upper bound: 0.0794123
time: 0.23 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0783930, upper bound: 0.0794123
time: 0.24 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0322431, 0.1248830, 0.0319700, 0.1270105, -0.0947674, 0.0929129
1: -0.0128046, 0.0114079, -0.0133929, 0.0117421, -0.0245467, 0.0248008
2: 0.0283025, 0.0518080, 0.0280271, 0.0521042, -0.0238017, 0.0237809
3: -0.0191366, -0.0029256, -0.0194361, -0.0026731, -0.0164635, 0.0165105
4: 0.0246338, 0.0467418, 0.0244007, 0.0470015, -0.0223677, 0.0223410

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0781469, upper bound: 0.0780590
time: 0.25 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0781469, upper bound: 0.0781955
time: 0.24 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0309500, 0.1755762, 0.0319700, 0.1270105, -0.0960605, 0.1436061
1: -0.0162868, 0.0147361, -0.0133929, 0.0117421, -0.0280289, 0.0281290
2: 0.0267344, 0.0555944, 0.0280271, 0.0521042, -0.0253698, 0.0275672
3: -0.0199336, 0.0016009, -0.0194361, -0.0026731, -0.0172605, 0.0210370
4: 0.0234237, 0.0507124, 0.0244007, 0.0470015, -0.0235778, 0.0263117

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0782377, upper bound: 0.0781715
time: 0.23 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0781574
time: 0.23 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.34 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0767145, upper bound: 0.0767145
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0767145, upper bound: 0.0767639
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0767639, upper bound: 0.0767145
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0767639, upper bound: 0.0767639
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0802910, upper bound: 0.0799221
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0803042, upper bound: 0.0798344
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0794123, upper bound: 0.0783930
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0794123, upper bound: 0.0783930
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0799221, upper bound: 0.0802910
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0798344, upper bound: 0.0803042
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0783930, upper bound: 0.0794123
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0783930, upper bound: 0.0794123
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0781469, upper bound: 0.0780590
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0781469, upper bound: 0.0781955
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0782377, upper bound: 0.0781715
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0781574

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0373425, 0.0757578, 0.0373425, 0.0757578, -0.0384154, 0.0384154
1: -0.0053885, 0.0112061, -0.0053885, 0.0112061, -0.0165946, 0.0165946
2: 0.0331099, 0.0513776, 0.0331099, 0.0513776, -0.0182677, 0.0182677
3: -0.0175277, -0.0075475, -0.0175277, -0.0075475, -0.0099802, 0.0099802
4: 0.0279520, 0.0458557, 0.0279520, 0.0458557, -0.0179037, 0.0179037

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0766690, upper bound: 0.0796515
time: 0.24 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0767145, upper bound: 0.0802488
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0373425, 0.0757578, 0.0361534, 0.1192177, -0.0818753, 0.0396045
1: -0.0053885, 0.0112061, -0.0082976, 0.0143075, -0.0196960, 0.0195037
2: 0.0331099, 0.0513776, 0.0316524, 0.0546903, -0.0215803, 0.0197252
3: -0.0175277, -0.0075475, -0.0182268, -0.0039084, -0.0136193, 0.0106793
4: 0.0279520, 0.0458557, 0.0268038, 0.0493851, -0.0214331, 0.0190519

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0756634, upper bound: 0.0784612
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0766690, upper bound: 0.0796955
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0767145, upper bound: 0.0802928
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0361534, 0.1192177, 0.0373425, 0.0757578, -0.0396045, 0.0818753
1: -0.0082976, 0.0143075, -0.0053885, 0.0112061, -0.0195037, 0.0196960
2: 0.0316524, 0.0546903, 0.0331099, 0.0513776, -0.0197252, 0.0215803
3: -0.0182268, -0.0039084, -0.0175277, -0.0075475, -0.0106793, 0.0136193
4: 0.0268038, 0.0493851, 0.0279520, 0.0458557, -0.0190519, 0.0214331

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0764173, upper bound: 0.0766641
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0767586, upper bound: 0.0767145
time: 0.26 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0361534, 0.1192177, 0.0361534, 0.1192177, -0.0830644, 0.0830644
1: -0.0082976, 0.0143075, -0.0082976, 0.0143075, -0.0226051, 0.0226051
2: 0.0316524, 0.0546903, 0.0316524, 0.0546903, -0.0230379, 0.0230379
3: -0.0182268, -0.0039084, -0.0182268, -0.0039084, -0.0143185, 0.0143185
4: 0.0268038, 0.0493851, 0.0268038, 0.0493851, -0.0225813, 0.0225813

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0766982, upper bound: 0.0763732
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0767586, upper bound: 0.0767145
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0372057, 0.0714706, 0.0322136, 0.1161338, -0.0789281, 0.0392570
1: -0.0056048, 0.0110969, -0.0126322, 0.0111528, -0.0167577, 0.0237291
2: 0.0330311, 0.0513681, 0.0284116, 0.0517276, -0.0186965, 0.0229565
3: -0.0177317, -0.0077312, -0.0191963, -0.0035721, -0.0141596, 0.0114651
4: 0.0279229, 0.0458125, 0.0247197, 0.0465812, -0.0186583, 0.0210929

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0802549, upper bound: 0.0792679
time: 0.23 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0802549, upper bound: 0.0798344
time: 0.23 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0372057, 0.0714706, 0.0320979, 0.1254924, -0.0882867, 0.0393727
1: -0.0056048, 0.0110969, -0.0131760, 0.0114695, -0.0170743, 0.0242729
2: 0.0330311, 0.0513681, 0.0282175, 0.0518427, -0.0188117, 0.0231506
3: -0.0177317, -0.0077312, -0.0193456, -0.0028562, -0.0148755, 0.0116144
4: 0.0279229, 0.0458125, 0.0245342, 0.0467785, -0.0188556, 0.0212783

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0802680, upper bound: 0.0792679
time: 0.23 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0802680, upper bound: 0.0798344
time: 0.23 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0371996, 0.0817597, 0.0321587, 0.1197090, -0.0825095, 0.0496010
1: -0.0060085, 0.0113276, -0.0128977, 0.0114847, -0.0174932, 0.0242253
2: 0.0329229, 0.0514867, 0.0282688, 0.0518810, -0.0189581, 0.0232179
3: -0.0178054, -0.0073031, -0.0193458, -0.0032745, -0.0145309, 0.0120428
4: 0.0277720, 0.0460424, 0.0246230, 0.0467204, -0.0189484, 0.0214194

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 28

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0371996, 0.0817597, 0.0322635, 0.1274086, -0.0902090, 0.0494962
1: -0.0060085, 0.0113276, -0.0131250, 0.0116280, -0.0176365, 0.0244526
2: 0.0329229, 0.0514867, 0.0282197, 0.0519095, -0.0189866, 0.0232670
3: -0.0178054, -0.0073031, -0.0193775, -0.0026727, -0.0151326, 0.0120744
4: 0.0277720, 0.0460424, 0.0245329, 0.0468749, -0.0191029, 0.0215095

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 28

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0322136, 0.1161338, 0.0372057, 0.0714706, -0.0392570, 0.0789281
1: -0.0126322, 0.0111528, -0.0056048, 0.0110969, -0.0237291, 0.0167577
2: 0.0284116, 0.0517276, 0.0330311, 0.0513681, -0.0229565, 0.0186965
3: -0.0191963, -0.0035721, -0.0177317, -0.0077312, -0.0114651, 0.0141596
4: 0.0247197, 0.0465812, 0.0279229, 0.0458125, -0.0210929, 0.0186583

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0792679, upper bound: 0.0802549
time: 0.23 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0792679, upper bound: 0.0802910
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0320979, 0.1254924, 0.0372057, 0.0714706, -0.0393727, 0.0882867
1: -0.0131760, 0.0114695, -0.0056048, 0.0110969, -0.0242729, 0.0170743
2: 0.0282175, 0.0518427, 0.0330311, 0.0513681, -0.0231506, 0.0188117
3: -0.0193456, -0.0028562, -0.0177317, -0.0077312, -0.0116144, 0.0148755
4: 0.0245342, 0.0467785, 0.0279229, 0.0458125, -0.0212783, 0.0188556

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0792679, upper bound: 0.0802680
time: 0.24 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0792679, upper bound: 0.0803042
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0321587, 0.1197090, 0.0371996, 0.0817597, -0.0496010, 0.0825095
1: -0.0128977, 0.0114847, -0.0060085, 0.0113276, -0.0242253, 0.0174932
2: 0.0282688, 0.0518810, 0.0329229, 0.0514867, -0.0232179, 0.0189581
3: -0.0193458, -0.0032745, -0.0178054, -0.0073031, -0.0120428, 0.0145309
4: 0.0246230, 0.0467204, 0.0277720, 0.0460424, -0.0214194, 0.0189484

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 28

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0322635, 0.1274086, 0.0371996, 0.0817597, -0.0494962, 0.0902090
1: -0.0131250, 0.0116280, -0.0060085, 0.0113276, -0.0244526, 0.0176365
2: 0.0282197, 0.0519095, 0.0329229, 0.0514867, -0.0232670, 0.0189866
3: -0.0193775, -0.0026727, -0.0178054, -0.0073031, -0.0120744, 0.0151326
4: 0.0245329, 0.0468749, 0.0277720, 0.0460424, -0.0215095, 0.0191029

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 28

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0322431, 0.1248830, 0.0322431, 0.1248830, -0.0926399, 0.0926399
1: -0.0128046, 0.0114079, -0.0128046, 0.0114079, -0.0242124, 0.0242124
2: 0.0283025, 0.0518080, 0.0283025, 0.0518080, -0.0235055, 0.0235055
3: -0.0191366, -0.0029256, -0.0191366, -0.0029256, -0.0162110, 0.0162110
4: 0.0246338, 0.0467418, 0.0246338, 0.0467418, -0.0221080, 0.0221080

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0781454, upper bound: 0.0805235
time: 0.24 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0780385, upper bound: 0.0805554
time: 0.24 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0322431, 0.1248830, 0.0309500, 0.1755762, -0.1433331, 0.0939330
1: -0.0128046, 0.0114079, -0.0162868, 0.0147361, -0.0275406, 0.0276946
2: 0.0283025, 0.0518080, 0.0267344, 0.0555944, -0.0272918, 0.0250736
3: -0.0191366, -0.0029256, -0.0199336, 0.0016009, -0.0207375, 0.0170080
4: 0.0246338, 0.0467418, 0.0234237, 0.0507124, -0.0260786, 0.0233181

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0781454, upper bound: 0.0806378
time: 0.24 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0780385, upper bound: 0.0806678
time: 0.24 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0309500, 0.1755762, 0.0322136, 0.1161338, -0.0851839, 0.1433626
1: -0.0162868, 0.0147361, -0.0126322, 0.0111528, -0.0274396, 0.0273683
2: 0.0267344, 0.0555944, 0.0284116, 0.0517276, -0.0249932, 0.0271827
3: -0.0199336, 0.0016009, -0.0191963, -0.0035721, -0.0163615, 0.0207972
4: 0.0234237, 0.0507124, 0.0247197, 0.0465812, -0.0231575, 0.0259928

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0781574
time: 0.24 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0781574
time: 0.24 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0309500, 0.1755762, 0.0320979, 0.1254924, -0.0945425, 0.1434783
1: -0.0162868, 0.0147361, -0.0131760, 0.0114695, -0.0277562, 0.0279121
2: 0.0267344, 0.0555944, 0.0282175, 0.0518427, -0.0251084, 0.0273768
3: -0.0199336, 0.0016009, -0.0193456, -0.0028562, -0.0170774, 0.0209465
4: 0.0234237, 0.0507124, 0.0245342, 0.0467785, -0.0233548, 0.0261782

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B2_B1

### Relational analysis result of NS_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0780186
time: 0.24 seconds

## Relational analysis of NS_A2_B2_A2_B2_B2

### Relational analysis result of NS_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0780186
time: 0.24 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.76 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 0, lower bound: -0.0766690, upper bound: 0.0796515
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 0, lower bound: -0.0767145, upper bound: 0.0802488
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 0, lower bound: -0.0766690, upper bound: 0.0796955
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 0, lower bound: -0.0767145, upper bound: 0.0802928
NS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 0, lower bound: -0.0764173, upper bound: 0.0766641
NS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 0, lower bound: -0.0767586, upper bound: 0.0767145
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 0, lower bound: -0.0766982, upper bound: 0.0763732
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 0, lower bound: -0.0767586, upper bound: 0.0767145
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 0, lower bound: -0.0802549, upper bound: 0.0792679
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 0, lower bound: -0.0802549, upper bound: 0.0798344
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 0, lower bound: -0.0802680, upper bound: 0.0792679
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 0, lower bound: -0.0802680, upper bound: 0.0798344
NS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 0, lower bound: -0.0792679, upper bound: 0.0802549
NS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 0, lower bound: -0.0792679, upper bound: 0.0802910
NS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 0, lower bound: -0.0792679, upper bound: 0.0802680
NS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 0, lower bound: -0.0792679, upper bound: 0.0803042
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 0, lower bound: -0.0781454, upper bound: 0.0805235
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 0, lower bound: -0.0780385, upper bound: 0.0805554
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 0, lower bound: -0.0781454, upper bound: 0.0806378
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 0, lower bound: -0.0780385, upper bound: 0.0806678
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0781574
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0781574
NS_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0780186
NS_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.76
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0780186

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0374793, 0.0656538, 0.0373425, 0.0757578, -0.0382785, 0.0283114
1: -0.0049000, 0.0105330, -0.0053885, 0.0112061, -0.0161061, 0.0159215
2: 0.0333420, 0.0507765, 0.0331099, 0.0513776, -0.0180356, 0.0176666
3: -0.0173014, -0.0082131, -0.0175277, -0.0075475, -0.0097539, 0.0093146
4: 0.0281858, 0.0452273, 0.0279520, 0.0458557, -0.0176699, 0.0172753

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0796277, upper bound: 0.0796277
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0796277, upper bound: 0.0796515
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0374077, 0.0746413, 0.0373425, 0.0757578, -0.0383501, 0.0372989
1: -0.0052945, 0.0110442, -0.0053885, 0.0112061, -0.0165006, 0.0164327
2: 0.0331874, 0.0511943, 0.0331099, 0.0513776, -0.0181902, 0.0180843
3: -0.0174662, -0.0076531, -0.0175277, -0.0075475, -0.0099187, 0.0098746
4: 0.0280150, 0.0457055, 0.0279520, 0.0458557, -0.0178407, 0.0177535

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0796515, upper bound: 0.0802250
time: 0.24 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0796515, upper bound: 0.0802488
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0374793, 0.0656538, 0.0361534, 0.1192177, -0.0817384, 0.0295005
1: -0.0049000, 0.0105330, -0.0082976, 0.0143075, -0.0192075, 0.0188306
2: 0.0333420, 0.0507765, 0.0316524, 0.0546903, -0.0213483, 0.0191241
3: -0.0173014, -0.0082131, -0.0182268, -0.0039084, -0.0133930, 0.0100137
4: 0.0281858, 0.0452273, 0.0268038, 0.0493851, -0.0211992, 0.0184236

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0763494, upper bound: 0.0796560
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0763494, upper bound: 0.0796955
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0374077, 0.0746413, 0.0361534, 0.1192177, -0.0818100, 0.0384879
1: -0.0052945, 0.0110442, -0.0082976, 0.0143075, -0.0196020, 0.0193417
2: 0.0331874, 0.0511943, 0.0316524, 0.0546903, -0.0215029, 0.0195419
3: -0.0174662, -0.0076531, -0.0182268, -0.0039084, -0.0135578, 0.0105738
4: 0.0280150, 0.0457055, 0.0268038, 0.0493851, -0.0213700, 0.0189017

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0763732, upper bound: 0.0802533
time: 0.24 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0763732, upper bound: 0.0802533
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: 0.0361534, 0.1192177, 0.0374793, 0.0656538, -0.0295005, 0.0817384
1: -0.0082976, 0.0143075, -0.0049000, 0.0105330, -0.0188306, 0.0192075
2: 0.0316524, 0.0546903, 0.0333420, 0.0507765, -0.0191241, 0.0213483
3: -0.0182268, -0.0039084, -0.0173014, -0.0082131, -0.0100137, 0.0133930
4: 0.0268038, 0.0493851, 0.0281858, 0.0452273, -0.0184236, 0.0211992

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0796560, upper bound: 0.0763494
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0796560, upper bound: 0.0766690
time: 0.24 seconds

## BFS NS instance: NS_A1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: 0.0361534, 0.1192177, 0.0374077, 0.0746413, -0.0384879, 0.0818100
1: -0.0082976, 0.0143075, -0.0052945, 0.0110442, -0.0193417, 0.0196020
2: 0.0316524, 0.0546903, 0.0331874, 0.0511943, -0.0195419, 0.0215029
3: -0.0182268, -0.0039084, -0.0174662, -0.0076531, -0.0105738, 0.0135578
4: 0.0268038, 0.0493851, 0.0280150, 0.0457055, -0.0189017, 0.0213700

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0802533, upper bound: 0.0763732
time: 0.25 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0802533, upper bound: 0.0767145
time: 0.24 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0363850, 0.1089879, 0.0361534, 0.1192177, -0.0828327, 0.0728346
1: -0.0076837, 0.0136366, -0.0082976, 0.0143075, -0.0219912, 0.0219341
2: 0.0319655, 0.0541022, 0.0316524, 0.0546903, -0.0227248, 0.0224498
3: -0.0179909, -0.0049057, -0.0182268, -0.0039084, -0.0140825, 0.0133212
4: 0.0270599, 0.0487460, 0.0268038, 0.0493851, -0.0223252, 0.0219422

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0763778, upper bound: 0.0763440
time: 0.24 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0763778, upper bound: 0.0763732
time: 0.24 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0362448, 0.1181611, 0.0361534, 0.1192177, -0.0829729, 0.0820078
1: -0.0081737, 0.0141474, -0.0082976, 0.0143075, -0.0224812, 0.0224450
2: 0.0317618, 0.0545166, 0.0316524, 0.0546903, -0.0229285, 0.0228642
3: -0.0181626, -0.0040423, -0.0182268, -0.0039084, -0.0142543, 0.0141845
4: 0.0268756, 0.0492348, 0.0268038, 0.0493851, -0.0225094, 0.0224311

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0764173, upper bound: 0.0766641
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0764173, upper bound: 0.0767145
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0373131, 0.0614454, 0.0322136, 0.1161338, -0.0788208, 0.0292318
1: -0.0051779, 0.0104210, -0.0126322, 0.0111528, -0.0163307, 0.0230532
2: 0.0332269, 0.0507907, 0.0284116, 0.0517276, -0.0185007, 0.0223790
3: -0.0175076, -0.0083929, -0.0191963, -0.0035721, -0.0139355, 0.0108034
4: 0.0281332, 0.0451879, 0.0247197, 0.0465812, -0.0184480, 0.0204682

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0801701, upper bound: 0.0795420
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0801701, upper bound: 0.0795420
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0372433, 0.0703479, 0.0322136, 0.1161338, -0.0788905, 0.0381344
1: -0.0055427, 0.0109307, -0.0126322, 0.0111528, -0.0166956, 0.0235629
2: 0.0330849, 0.0511877, 0.0284116, 0.0517276, -0.0186427, 0.0227760
3: -0.0176648, -0.0078391, -0.0191963, -0.0035721, -0.0140927, 0.0113572
4: 0.0279738, 0.0456585, 0.0247197, 0.0465812, -0.0186074, 0.0209389

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0801701, upper bound: 0.0799221
time: 0.25 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0801701, upper bound: 0.0799221
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0373131, 0.0614454, 0.0320979, 0.1254924, -0.0881794, 0.0293475
1: -0.0051779, 0.0104210, -0.0131760, 0.0114695, -0.0166474, 0.0235971
2: 0.0332269, 0.0507907, 0.0282175, 0.0518427, -0.0186158, 0.0225731
3: -0.0175076, -0.0083929, -0.0193456, -0.0028562, -0.0146514, 0.0109527
4: 0.0281332, 0.0451879, 0.0245342, 0.0467785, -0.0186453, 0.0206537

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0801701, upper bound: 0.0792679
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0801701, upper bound: 0.0792679
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0372433, 0.0703479, 0.0320979, 0.1254924, -0.0882492, 0.0382500
1: -0.0055427, 0.0109307, -0.0131760, 0.0114695, -0.0170122, 0.0241067
2: 0.0330849, 0.0511877, 0.0282175, 0.0518427, -0.0187578, 0.0229701
3: -0.0176648, -0.0078391, -0.0193456, -0.0028562, -0.0148086, 0.0115065
4: 0.0279738, 0.0456585, 0.0245342, 0.0467785, -0.0188046, 0.0211243

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0801701, upper bound: 0.0792679
time: 0.25 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0801701, upper bound: 0.0798344
time: 0.26 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0322136, 0.1161338, 0.0373131, 0.0614454, -0.0292318, 0.0788208
1: -0.0126322, 0.0111528, -0.0051779, 0.0104210, -0.0230532, 0.0163307
2: 0.0284116, 0.0517276, 0.0332269, 0.0507907, -0.0223790, 0.0185007
3: -0.0191963, -0.0035721, -0.0175076, -0.0083929, -0.0108034, 0.0139355
4: 0.0247197, 0.0465812, 0.0281332, 0.0451879, -0.0204682, 0.0184480

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0795420, upper bound: 0.0801701
time: 0.25 seconds

## Relational analysis of NS_A2_B1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0795420, upper bound: 0.0802549
time: 0.23 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0322136, 0.1161338, 0.0372433, 0.0703479, -0.0381344, 0.0788905
1: -0.0126322, 0.0111528, -0.0055427, 0.0109307, -0.0235629, 0.0166956
2: 0.0284116, 0.0517276, 0.0330849, 0.0511877, -0.0227760, 0.0186427
3: -0.0191963, -0.0035721, -0.0176648, -0.0078391, -0.0113572, 0.0140927
4: 0.0247197, 0.0465812, 0.0279738, 0.0456585, -0.0209389, 0.0186074

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0795420, upper bound: 0.0802062
time: 0.24 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0795420, upper bound: 0.0802910
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0320979, 0.1254924, 0.0373131, 0.0614454, -0.0293475, 0.0881794
1: -0.0131760, 0.0114695, -0.0051779, 0.0104210, -0.0235971, 0.0166474
2: 0.0282175, 0.0518427, 0.0332269, 0.0507907, -0.0225731, 0.0186158
3: -0.0193456, -0.0028562, -0.0175076, -0.0083929, -0.0109527, 0.0146514
4: 0.0245342, 0.0467785, 0.0281332, 0.0451879, -0.0206537, 0.0186453

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0792679, upper bound: 0.0802581
time: 0.25 seconds

## Relational analysis of NS_A2_B1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0792679, upper bound: 0.0802680
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0320979, 0.1254924, 0.0372433, 0.0703479, -0.0382500, 0.0882492
1: -0.0131760, 0.0114695, -0.0055427, 0.0109307, -0.0241067, 0.0170122
2: 0.0282175, 0.0518427, 0.0330849, 0.0511877, -0.0229701, 0.0187578
3: -0.0193456, -0.0028562, -0.0176648, -0.0078391, -0.0115065, 0.0148086
4: 0.0245342, 0.0467785, 0.0279738, 0.0456585, -0.0211243, 0.0188046

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0792679, upper bound: 0.0802790
time: 0.23 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0792679, upper bound: 0.0802849
time: 0.24 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0324816, 0.1140448, 0.0322431, 0.1248830, -0.0924013, 0.0818017
1: -0.0120535, 0.0108358, -0.0128046, 0.0114079, -0.0234614, 0.0236404
2: 0.0286829, 0.0514534, 0.0283025, 0.0518080, -0.0231251, 0.0231509
3: -0.0189003, -0.0038204, -0.0191366, -0.0029256, -0.0159747, 0.0153163
4: 0.0249442, 0.0463378, 0.0246338, 0.0467418, -0.0217975, 0.0217040

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0805799, upper bound: 0.0805235
time: 0.25 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0805799, upper bound: 0.0805235
time: 0.25 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0323709, 0.1233958, 0.0322431, 0.1248830, -0.0925120, 0.0911527
1: -0.0125881, 0.0111360, -0.0128046, 0.0114079, -0.0239960, 0.0239405
2: 0.0284928, 0.0515530, 0.0283025, 0.0518080, -0.0233152, 0.0232505
3: -0.0190458, -0.0031060, -0.0191366, -0.0029256, -0.0161202, 0.0160306
4: 0.0247660, 0.0465239, 0.0246338, 0.0467418, -0.0219758, 0.0218901

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0805799, upper bound: 0.0805554
time: 0.25 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0805799, upper bound: 0.0805554
time: 0.24 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0324816, 0.1140448, 0.0309500, 0.1755762, -0.1430945, 0.0830949
1: -0.0120535, 0.0108358, -0.0162868, 0.0147361, -0.0267895, 0.0271226
2: 0.0286829, 0.0514534, 0.0267344, 0.0555944, -0.0269115, 0.0247191
3: -0.0189003, -0.0038204, -0.0199336, 0.0016009, -0.0205012, 0.0161133
4: 0.0249442, 0.0463378, 0.0234237, 0.0507124, -0.0257682, 0.0229141

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0780385, upper bound: 0.0806378
time: 0.25 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0780385, upper bound: 0.0806378
time: 0.26 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0323709, 0.1233958, 0.0309500, 0.1755762, -0.1432052, 0.0924459
1: -0.0125881, 0.0111360, -0.0162868, 0.0147361, -0.0273242, 0.0274227
2: 0.0284928, 0.0515530, 0.0267344, 0.0555944, -0.0271016, 0.0248186
3: -0.0190458, -0.0031060, -0.0199336, 0.0016009, -0.0206467, 0.0168276
4: 0.0247660, 0.0465239, 0.0234237, 0.0507124, -0.0259465, 0.0231002

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0772501, upper bound: 0.0805296
time: 0.25 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0780385, upper bound: 0.0806678
time: 0.26 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0780385, upper bound: 0.0806678
time: 0.25 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0312119, 0.1642383, 0.0322136, 0.1161338, -0.0849219, 0.1320247
1: -0.0154898, 0.0140534, -0.0126322, 0.0111528, -0.0266426, 0.0266856
2: 0.0271566, 0.0549670, 0.0284116, 0.0517276, -0.0245710, 0.0265553
3: -0.0196514, 0.0005930, -0.0191963, -0.0035721, -0.0160793, 0.0197893
4: 0.0237919, 0.0500420, 0.0247197, 0.0465812, -0.0227893, 0.0253223

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0782377, upper bound: 0.0780431
time: 0.26 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0782377, upper bound: 0.0780431
time: 0.26 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0310772, 0.1741992, 0.0322136, 0.1161338, -0.0850566, 0.1419856
1: -0.0160661, 0.0144748, -0.0126322, 0.0111528, -0.0272189, 0.0271069
2: 0.0269344, 0.0553142, 0.0284116, 0.0517276, -0.0247932, 0.0269026
3: -0.0198422, 0.0014262, -0.0191963, -0.0035721, -0.0162701, 0.0206225
4: 0.0235673, 0.0504758, 0.0247197, 0.0465812, -0.0230139, 0.0257562

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0782377, upper bound: 0.0780431
time: 0.25 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0782377, upper bound: 0.0780431
time: 0.25 seconds

## BFS NS instance: NS_A2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: 0.0309500, 0.1755762, 0.0323709, 0.1233958, -0.0924459, 0.1432052
1: -0.0162868, 0.0147361, -0.0125881, 0.0111360, -0.0274227, 0.0273242
2: 0.0267344, 0.0555944, 0.0284928, 0.0515530, -0.0248186, 0.0271016
3: -0.0199336, 0.0016009, -0.0190458, -0.0031060, -0.0168276, 0.0206467
4: 0.0234237, 0.0507124, 0.0247660, 0.0465239, -0.0231002, 0.0259465

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0780186
time: 0.26 seconds

## Relational analysis of NS_A2_B2_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0780186
time: 0.25 seconds

## BFS NS instance: NS_A2_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: 0.0309500, 0.1755762, 0.0310772, 0.1740944, -0.1431444, 0.1444989
1: -0.0162868, 0.0147361, -0.0160661, 0.0144619, -0.0307487, 0.0308021
2: 0.0267344, 0.0555944, 0.0269344, 0.0551495, -0.0284151, 0.0286600
3: -0.0199336, 0.0016009, -0.0198422, 0.0013388, -0.0212724, 0.0214431
4: 0.0234237, 0.0507124, 0.0235673, 0.0503064, -0.0268827, 0.0271451

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0780186
time: 0.24 seconds

## Relational analysis of NS_A2_B2_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0780186
time: 0.26 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.48 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0796277, upper bound: 0.0796277
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0796277, upper bound: 0.0796515
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0796515, upper bound: 0.0802250
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0796515, upper bound: 0.0802488
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0763494, upper bound: 0.0796560
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0763494, upper bound: 0.0796955
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0763732, upper bound: 0.0802533
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0763732, upper bound: 0.0802533
NS_A1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0796560, upper bound: 0.0763494
NS_A1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0796560, upper bound: 0.0766690
NS_A1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0802533, upper bound: 0.0763732
NS_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0802533, upper bound: 0.0767145
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0763778, upper bound: 0.0763440
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0763778, upper bound: 0.0763732
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0764173, upper bound: 0.0766641
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0764173, upper bound: 0.0767145
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0801701, upper bound: 0.0795420
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0801701, upper bound: 0.0795420
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0801701, upper bound: 0.0799221
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0801701, upper bound: 0.0799221
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0801701, upper bound: 0.0792679
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0801701, upper bound: 0.0792679
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0801701, upper bound: 0.0792679
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0801701, upper bound: 0.0798344
NS_A2_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0795420, upper bound: 0.0801701
NS_A2_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0795420, upper bound: 0.0802549
NS_A2_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0795420, upper bound: 0.0802062
NS_A2_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0795420, upper bound: 0.0802910
NS_A2_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0792679, upper bound: 0.0802581
NS_A2_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0792679, upper bound: 0.0802680
NS_A2_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0792679, upper bound: 0.0802790
NS_A2_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0792679, upper bound: 0.0802849
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0805799, upper bound: 0.0805235
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0805799, upper bound: 0.0805235
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0805799, upper bound: 0.0805554
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0805799, upper bound: 0.0805554
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0780385, upper bound: 0.0806378
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0780385, upper bound: 0.0806378
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0780385, upper bound: 0.0806678
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0780385, upper bound: 0.0806678
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0782377, upper bound: 0.0780431
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0782377, upper bound: 0.0780431
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0782377, upper bound: 0.0780431
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0782377, upper bound: 0.0780431
NS_A2_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0780186
NS_A2_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0780186
NS_A2_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0780186
NS_A2_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0782362, upper bound: 0.0780186

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0374793, 0.0656538, 0.0374793, 0.0656538, -0.0281745, 0.0281745
1: -0.0049000, 0.0105330, -0.0049000, 0.0105330, -0.0154330, 0.0154330
2: 0.0333420, 0.0507765, 0.0333420, 0.0507765, -0.0174345, 0.0174345
3: -0.0173014, -0.0082131, -0.0173014, -0.0082131, -0.0090883, 0.0090883
4: 0.0281858, 0.0452273, 0.0281858, 0.0452273, -0.0170415, 0.0170415

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0374793, 0.0656538, 0.0374077, 0.0746413, -0.0371620, 0.0282461
1: -0.0049000, 0.0105330, -0.0052945, 0.0110442, -0.0159441, 0.0158275
2: 0.0333420, 0.0507765, 0.0331874, 0.0511943, -0.0178523, 0.0175891
3: -0.0173014, -0.0082131, -0.0174662, -0.0076531, -0.0096484, 0.0092531
4: 0.0281858, 0.0452273, 0.0280150, 0.0457055, -0.0175196, 0.0172123

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0374077, 0.0746413, 0.0374793, 0.0656538, -0.0282461, 0.0371620
1: -0.0052945, 0.0110442, -0.0049000, 0.0105330, -0.0158275, 0.0159441
2: 0.0331874, 0.0511943, 0.0333420, 0.0507765, -0.0175891, 0.0178523
3: -0.0174662, -0.0076531, -0.0173014, -0.0082131, -0.0092531, 0.0096484
4: 0.0280150, 0.0457055, 0.0281858, 0.0452273, -0.0172123, 0.0175196

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0374077, 0.0746413, 0.0374077, 0.0746413, -0.0372336, 0.0372336
1: -0.0052945, 0.0110442, -0.0052945, 0.0110442, -0.0163386, 0.0163386
2: 0.0331874, 0.0511943, 0.0331874, 0.0511943, -0.0180069, 0.0180069
3: -0.0174662, -0.0076531, -0.0174662, -0.0076531, -0.0098131, 0.0098131
4: 0.0280150, 0.0457055, 0.0280150, 0.0457055, -0.0176905, 0.0176905

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0374793, 0.0656538, 0.0363850, 0.1089879, -0.0715086, 0.0292688
1: -0.0049000, 0.0105330, -0.0076837, 0.0136366, -0.0185365, 0.0182167
2: 0.0333420, 0.0507765, 0.0319655, 0.0541022, -0.0207602, 0.0188110
3: -0.0173014, -0.0082131, -0.0179909, -0.0049057, -0.0123958, 0.0097778
4: 0.0281858, 0.0452273, 0.0270599, 0.0487460, -0.0205601, 0.0181674

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0374793, 0.0656538, 0.0362448, 0.1181611, -0.0806818, 0.0294091
1: -0.0049000, 0.0105330, -0.0081737, 0.0141474, -0.0190474, 0.0187067
2: 0.0333420, 0.0507765, 0.0317618, 0.0545166, -0.0211746, 0.0190147
3: -0.0173014, -0.0082131, -0.0181626, -0.0040423, -0.0132591, 0.0099495
4: 0.0281858, 0.0452273, 0.0268756, 0.0492348, -0.0210490, 0.0183517

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0374077, 0.0746413, 0.0363850, 0.1089879, -0.0715802, 0.0382563
1: -0.0052945, 0.0110442, -0.0076837, 0.0136366, -0.0189310, 0.0187279
2: 0.0331874, 0.0511943, 0.0319655, 0.0541022, -0.0209148, 0.0192287
3: -0.0174662, -0.0076531, -0.0179909, -0.0049057, -0.0125605, 0.0103378
4: 0.0280150, 0.0457055, 0.0270599, 0.0487460, -0.0207309, 0.0186456

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0374077, 0.0746413, 0.0362448, 0.1181611, -0.0807534, 0.0383965
1: -0.0052945, 0.0110442, -0.0081737, 0.0141474, -0.0194419, 0.0192179
2: 0.0331874, 0.0511943, 0.0317618, 0.0545166, -0.0213291, 0.0194325
3: -0.0174662, -0.0076531, -0.0181626, -0.0040423, -0.0134239, 0.0105096
4: 0.0280150, 0.0457055, 0.0268756, 0.0492348, -0.0212198, 0.0188299

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0363850, 0.1089879, 0.0374793, 0.0656538, -0.0292688, 0.0715086
1: -0.0076837, 0.0136366, -0.0049000, 0.0105330, -0.0182167, 0.0185365
2: 0.0319655, 0.0541022, 0.0333420, 0.0507765, -0.0188110, 0.0207602
3: -0.0179909, -0.0049057, -0.0173014, -0.0082131, -0.0097778, 0.0123958
4: 0.0270599, 0.0487460, 0.0281858, 0.0452273, -0.0181674, 0.0205601

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0362448, 0.1181611, 0.0374793, 0.0656538, -0.0294091, 0.0806818
1: -0.0081737, 0.0141474, -0.0049000, 0.0105330, -0.0187067, 0.0190474
2: 0.0317618, 0.0545166, 0.0333420, 0.0507765, -0.0190147, 0.0211746
3: -0.0181626, -0.0040423, -0.0173014, -0.0082131, -0.0099495, 0.0132591
4: 0.0268756, 0.0492348, 0.0281858, 0.0452273, -0.0183517, 0.0210490

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0363850, 0.1089879, 0.0374077, 0.0746413, -0.0382563, 0.0715802
1: -0.0076837, 0.0136366, -0.0052945, 0.0110442, -0.0187279, 0.0189310
2: 0.0319655, 0.0541022, 0.0331874, 0.0511943, -0.0192287, 0.0209148
3: -0.0179909, -0.0049057, -0.0174662, -0.0076531, -0.0103378, 0.0125605
4: 0.0270599, 0.0487460, 0.0280150, 0.0457055, -0.0186456, 0.0207309

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0362448, 0.1181611, 0.0374077, 0.0746413, -0.0383965, 0.0807534
1: -0.0081737, 0.0141474, -0.0052945, 0.0110442, -0.0192179, 0.0194419
2: 0.0317618, 0.0545166, 0.0331874, 0.0511943, -0.0194325, 0.0213291
3: -0.0181626, -0.0040423, -0.0174662, -0.0076531, -0.0105096, 0.0134239
4: 0.0268756, 0.0492348, 0.0280150, 0.0457055, -0.0188299, 0.0212198

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0363850, 0.1089879, 0.0363850, 0.1089879, -0.0726029, 0.0726029
1: -0.0076837, 0.0136366, -0.0076837, 0.0136366, -0.0213203, 0.0213203
2: 0.0319655, 0.0541022, 0.0319655, 0.0541022, -0.0221367, 0.0221367
3: -0.0179909, -0.0049057, -0.0179909, -0.0049057, -0.0130852, 0.0130852
4: 0.0270599, 0.0487460, 0.0270599, 0.0487460, -0.0216861, 0.0216861

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0363850, 0.1089879, 0.0362448, 0.1181611, -0.0817761, 0.0727431
1: -0.0076837, 0.0136366, -0.0081737, 0.0141474, -0.0218311, 0.0218103
2: 0.0319655, 0.0541022, 0.0317618, 0.0545166, -0.0225510, 0.0223404
3: -0.0179909, -0.0049057, -0.0181626, -0.0040423, -0.0139486, 0.0132570
4: 0.0270599, 0.0487460, 0.0268756, 0.0492348, -0.0221749, 0.0218704

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0362448, 0.1181611, 0.0363850, 0.1089879, -0.0727431, 0.0817761
1: -0.0081737, 0.0141474, -0.0076837, 0.0136366, -0.0218103, 0.0218311
2: 0.0317618, 0.0545166, 0.0319655, 0.0541022, -0.0223404, 0.0225510
3: -0.0181626, -0.0040423, -0.0179909, -0.0049057, -0.0132570, 0.0139486
4: 0.0268756, 0.0492348, 0.0270599, 0.0487460, -0.0218704, 0.0221749

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0362448, 0.1181611, 0.0362448, 0.1181611, -0.0819164, 0.0819164
1: -0.0081737, 0.0141474, -0.0081737, 0.0141474, -0.0223211, 0.0223211
2: 0.0317618, 0.0545166, 0.0317618, 0.0545166, -0.0227548, 0.0227548
3: -0.0181626, -0.0040423, -0.0181626, -0.0040423, -0.0141203, 0.0141203
4: 0.0268756, 0.0492348, 0.0268756, 0.0492348, -0.0223592, 0.0223592

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0373131, 0.0614454, 0.0324001, 0.1088825, -0.0715694, 0.0290452
1: -0.0051779, 0.0104210, -0.0121443, 0.0109089, -0.0160868, 0.0225654
2: 0.0332269, 0.0507907, 0.0286495, 0.0515194, -0.0182924, 0.0221411
3: -0.0175076, -0.0083929, -0.0191105, -0.0041677, -0.0133399, 0.0107176
4: 0.0281332, 0.0451879, 0.0249324, 0.0463131, -0.0181799, 0.0202555

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0373131, 0.0614454, 0.0325448, 0.1163658, -0.0790528, 0.0289006
1: -0.0051779, 0.0104210, -0.0123080, 0.0109378, -0.0161157, 0.0227290
2: 0.0332269, 0.0507907, 0.0286701, 0.0514502, -0.0182233, 0.0221206
3: -0.0175076, -0.0083929, -0.0190968, -0.0036137, -0.0138940, 0.0107039
4: 0.0281332, 0.0451879, 0.0248934, 0.0463973, -0.0182641, 0.0202945

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0372433, 0.0703479, 0.0324001, 0.1088825, -0.0716392, 0.0379478
1: -0.0055427, 0.0109307, -0.0121443, 0.0109089, -0.0164517, 0.0230750
2: 0.0330849, 0.0511877, 0.0286495, 0.0515194, -0.0184344, 0.0225381
3: -0.0176648, -0.0078391, -0.0191105, -0.0041677, -0.0134971, 0.0112714
4: 0.0279738, 0.0456585, 0.0249324, 0.0463131, -0.0183392, 0.0207262

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0372433, 0.0703479, 0.0325448, 0.1163658, -0.0791226, 0.0378032
1: -0.0055427, 0.0109307, -0.0123080, 0.0109378, -0.0164806, 0.0232387
2: 0.0330849, 0.0511877, 0.0286701, 0.0514502, -0.0183653, 0.0225176
3: -0.0176648, -0.0078391, -0.0190968, -0.0036137, -0.0140511, 0.0112577
4: 0.0279738, 0.0456585, 0.0248934, 0.0463973, -0.0184235, 0.0207651

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0373131, 0.0614454, 0.0322884, 0.1181857, -0.0808726, 0.0291570
1: -0.0051779, 0.0104210, -0.0126812, 0.0112108, -0.0163887, 0.0231022
2: 0.0332269, 0.0507907, 0.0284608, 0.0516205, -0.0183936, 0.0223299
3: -0.0175076, -0.0083929, -0.0192550, -0.0034590, -0.0140486, 0.0108621
4: 0.0281332, 0.0451879, 0.0247536, 0.0464990, -0.0183657, 0.0204343

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0373131, 0.0614454, 0.0323975, 0.1259867, -0.0886736, 0.0290479
1: -0.0051779, 0.0104210, -0.0129075, 0.0113624, -0.0165403, 0.0233285
2: 0.0332269, 0.0507907, 0.0284159, 0.0516519, -0.0184249, 0.0223747
3: -0.0175076, -0.0083929, -0.0192902, -0.0028489, -0.0146587, 0.0108973
4: 0.0281332, 0.0451879, 0.0246665, 0.0466550, -0.0185218, 0.0205214

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0372433, 0.0703479, 0.0322884, 0.1181857, -0.0809424, 0.0380596
1: -0.0055427, 0.0109307, -0.0126812, 0.0112108, -0.0167535, 0.0236119
2: 0.0330849, 0.0511877, 0.0284608, 0.0516205, -0.0185356, 0.0227269
3: -0.0176648, -0.0078391, -0.0192550, -0.0034590, -0.0142058, 0.0114159
4: 0.0279738, 0.0456585, 0.0247536, 0.0464990, -0.0185251, 0.0209050

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0372433, 0.0703479, 0.0323975, 0.1259867, -0.0887434, 0.0379505
1: -0.0055427, 0.0109307, -0.0129075, 0.0113624, -0.0169052, 0.0238382
2: 0.0330849, 0.0511877, 0.0284159, 0.0516519, -0.0185670, 0.0227717
3: -0.0176648, -0.0078391, -0.0192902, -0.0028489, -0.0148159, 0.0114511
4: 0.0279738, 0.0456585, 0.0246665, 0.0466550, -0.0186812, 0.0209921

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

## BFS NS instance: NS_A2_B1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0324001, 0.1088825, 0.0373131, 0.0614454, -0.0290452, 0.0715694
1: -0.0121443, 0.0109089, -0.0051779, 0.0104210, -0.0225654, 0.0160868
2: 0.0286495, 0.0515194, 0.0332269, 0.0507907, -0.0221411, 0.0182924
3: -0.0191105, -0.0041677, -0.0175076, -0.0083929, -0.0107176, 0.0133399
4: 0.0249324, 0.0463131, 0.0281332, 0.0451879, -0.0202555, 0.0181799

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

## BFS NS instance: NS_A2_B1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.0325448, 0.1163658, 0.0373131, 0.0614454, -0.0289006, 0.0790528
1: -0.0123080, 0.0109378, -0.0051779, 0.0104210, -0.0227290, 0.0161157
2: 0.0286701, 0.0514502, 0.0332269, 0.0507907, -0.0221206, 0.0182233
3: -0.0190968, -0.0036137, -0.0175076, -0.0083929, -0.0107039, 0.0138940
4: 0.0248934, 0.0463973, 0.0281332, 0.0451879, -0.0202945, 0.0182641

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

## BFS NS instance: NS_A2_B1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0324001, 0.1088825, 0.0372433, 0.0703479, -0.0379478, 0.0716392
1: -0.0121443, 0.0109089, -0.0055427, 0.0109307, -0.0230750, 0.0164517
2: 0.0286495, 0.0515194, 0.0330849, 0.0511877, -0.0225381, 0.0184344
3: -0.0191105, -0.0041677, -0.0176648, -0.0078391, -0.0112714, 0.0134971
4: 0.0249324, 0.0463131, 0.0279738, 0.0456585, -0.0207262, 0.0183392

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

## BFS NS instance: NS_A2_B1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.0325448, 0.1163658, 0.0372433, 0.0703479, -0.0378032, 0.0791226
1: -0.0123080, 0.0109378, -0.0055427, 0.0109307, -0.0232387, 0.0164806
2: 0.0286701, 0.0514502, 0.0330849, 0.0511877, -0.0225176, 0.0183653
3: -0.0190968, -0.0036137, -0.0176648, -0.0078391, -0.0112577, 0.0140511
4: 0.0248934, 0.0463973, 0.0279738, 0.0456585, -0.0207651, 0.0184235

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

## BFS NS instance: NS_A2_B1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0322884, 0.1181857, 0.0373131, 0.0614454, -0.0291570, 0.0808726
1: -0.0126812, 0.0112108, -0.0051779, 0.0104210, -0.0231022, 0.0163887
2: 0.0284608, 0.0516205, 0.0332269, 0.0507907, -0.0223299, 0.0183936
3: -0.0192550, -0.0034590, -0.0175076, -0.0083929, -0.0108621, 0.0140486
4: 0.0247536, 0.0464990, 0.0281332, 0.0451879, -0.0204343, 0.0183657

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

## BFS NS instance: NS_A2_B1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0323975, 0.1259867, 0.0373131, 0.0614454, -0.0290479, 0.0886736
1: -0.0129075, 0.0113624, -0.0051779, 0.0104210, -0.0233285, 0.0165403
2: 0.0284159, 0.0516519, 0.0332269, 0.0507907, -0.0223747, 0.0184249
3: -0.0192902, -0.0028489, -0.0175076, -0.0083929, -0.0108973, 0.0146587
4: 0.0246665, 0.0466550, 0.0281332, 0.0451879, -0.0205214, 0.0185218

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

## BFS NS instance: NS_A2_B1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0322884, 0.1181857, 0.0372433, 0.0703479, -0.0380596, 0.0809424
1: -0.0126812, 0.0112108, -0.0055427, 0.0109307, -0.0236119, 0.0167535
2: 0.0284608, 0.0516205, 0.0330849, 0.0511877, -0.0227269, 0.0185356
3: -0.0192550, -0.0034590, -0.0176648, -0.0078391, -0.0114159, 0.0142058
4: 0.0247536, 0.0464990, 0.0279738, 0.0456585, -0.0209050, 0.0185251

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

## BFS NS instance: NS_A2_B1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0323975, 0.1259867, 0.0372433, 0.0703479, -0.0379505, 0.0887434
1: -0.0129075, 0.0113624, -0.0055427, 0.0109307, -0.0238382, 0.0169052
2: 0.0284159, 0.0516519, 0.0330849, 0.0511877, -0.0227717, 0.0185670
3: -0.0192902, -0.0028489, -0.0176648, -0.0078391, -0.0114511, 0.0148159
4: 0.0246665, 0.0466550, 0.0279738, 0.0456585, -0.0209921, 0.0186812

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0324816, 0.1140448, 0.0324816, 0.1140448, -0.0815632, 0.0815632
1: -0.0120535, 0.0108358, -0.0120535, 0.0108358, -0.0228893, 0.0228893
2: 0.0286829, 0.0514534, 0.0286829, 0.0514534, -0.0227705, 0.0227705
3: -0.0189003, -0.0038204, -0.0189003, -0.0038204, -0.0150800, 0.0150800
4: 0.0249442, 0.0463378, 0.0249442, 0.0463378, -0.0213936, 0.0213936

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0805561, upper bound: 0.0797013
time: 0.26 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0800449, upper bound: 0.0797908
time: 0.25 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0324816, 0.1140448, 0.0323709, 0.1233958, -0.0909142, 0.0816739
1: -0.0120535, 0.0108358, -0.0125881, 0.0111360, -0.0231895, 0.0234240
2: 0.0286829, 0.0514534, 0.0284928, 0.0515530, -0.0228701, 0.0229607
3: -0.0189003, -0.0038204, -0.0190458, -0.0031060, -0.0157943, 0.0152254
4: 0.0249442, 0.0463378, 0.0247660, 0.0465239, -0.0215797, 0.0215718

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0799690, upper bound: 0.0803977
time: 0.26 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0800449, upper bound: 0.0797908
time: 0.25 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0323709, 0.1233958, 0.0324816, 0.1140448, -0.0816739, 0.0909142
1: -0.0125881, 0.0111360, -0.0120535, 0.0108358, -0.0234240, 0.0231895
2: 0.0284928, 0.0515530, 0.0286829, 0.0514534, -0.0229607, 0.0228701
3: -0.0190458, -0.0031060, -0.0189003, -0.0038204, -0.0152254, 0.0157943
4: 0.0247660, 0.0465239, 0.0249442, 0.0463378, -0.0215718, 0.0215797

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0804520, upper bound: 0.0797942
time: 0.26 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0797860, upper bound: 0.0798086
time: 0.25 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0323709, 0.1233958, 0.0323709, 0.1233958, -0.0910249, 0.0910249
1: -0.0125881, 0.0111360, -0.0125881, 0.0111360, -0.0237241, 0.0237241
2: 0.0284928, 0.0515530, 0.0284928, 0.0515530, -0.0230602, 0.0230602
3: -0.0190458, -0.0031060, -0.0190458, -0.0031060, -0.0159398, 0.0159398
4: 0.0247660, 0.0465239, 0.0247660, 0.0465239, -0.0217579, 0.0217579

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0804520, upper bound: 0.0797942
time: 0.26 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0797860, upper bound: 0.0798086
time: 0.26 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.0324816, 0.1140448, 0.0312119, 0.1642383, -0.1317566, 0.0828329
1: -0.0120535, 0.0108358, -0.0154898, 0.0140534, -0.0261068, 0.0263256
2: 0.0286829, 0.0514534, 0.0271566, 0.0549670, -0.0262841, 0.0242969
3: -0.0189003, -0.0038204, -0.0196514, 0.0005930, -0.0194933, 0.0158310
4: 0.0249442, 0.0463378, 0.0237919, 0.0500420, -0.0250978, 0.0225459

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0775215, upper bound: 0.0805186
time: 0.26 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.0324816, 0.1140448, 0.0310772, 0.1741992, -0.1417176, 0.0829676
1: -0.0120535, 0.0108358, -0.0160661, 0.0144748, -0.0265282, 0.0269019
2: 0.0286829, 0.0514534, 0.0269344, 0.0553142, -0.0266313, 0.0245190
3: -0.0189003, -0.0038204, -0.0198422, 0.0014262, -0.0203265, 0.0160218
4: 0.0249442, 0.0463378, 0.0235673, 0.0504758, -0.0255316, 0.0227705

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0775215, upper bound: 0.0805186
time: 0.26 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.0323709, 0.1233958, 0.0312119, 0.1642383, -0.1318673, 0.0921839
1: -0.0125881, 0.0111360, -0.0154898, 0.0140534, -0.0266415, 0.0266258
2: 0.0284928, 0.0515530, 0.0271566, 0.0549670, -0.0264742, 0.0243964
3: -0.0190458, -0.0031060, -0.0196514, 0.0005930, -0.0196388, 0.0165454
4: 0.0247660, 0.0465239, 0.0237919, 0.0500420, -0.0252760, 0.0227320

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.0323709, 0.1233958, 0.0310772, 0.1741992, -0.1418282, 0.0923186
1: -0.0125881, 0.0111360, -0.0160661, 0.0144748, -0.0270629, 0.0272021
2: 0.0284928, 0.0515530, 0.0269344, 0.0553142, -0.0268214, 0.0246186
3: -0.0190458, -0.0031060, -0.0198422, 0.0014262, -0.0204720, 0.0167362
4: 0.0247660, 0.0465239, 0.0235673, 0.0504758, -0.0257099, 0.0229566

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.0312119, 0.1642383, 0.0324816, 0.1140448, -0.0828329, 0.1317566
1: -0.0154898, 0.0140534, -0.0120535, 0.0108358, -0.0263256, 0.0261068
2: 0.0271566, 0.0549670, 0.0286829, 0.0514534, -0.0242969, 0.0262841
3: -0.0196514, 0.0005930, -0.0189003, -0.0038204, -0.0158310, 0.0194933
4: 0.0237919, 0.0500420, 0.0249442, 0.0463378, -0.0225459, 0.0250978

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.0312119, 0.1642383, 0.0312493, 0.1637233, -0.1325113, 0.1329890
1: -0.0154898, 0.0140534, -0.0154227, 0.0139402, -0.0294300, 0.0294761
2: 0.0271566, 0.0549670, 0.0272178, 0.0548256, -0.0276690, 0.0277492
3: -0.0196514, 0.0005930, -0.0196015, 0.0005169, -0.0201683, 0.0201945
4: 0.0237919, 0.0500420, 0.0238428, 0.0499271, -0.0261352, 0.0261992

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.0310772, 0.1741992, 0.0324816, 0.1140448, -0.0829676, 0.1417176
1: -0.0160661, 0.0144748, -0.0120535, 0.0108358, -0.0269019, 0.0265282
2: 0.0269344, 0.0553142, 0.0286829, 0.0514534, -0.0245190, 0.0266313
3: -0.0198422, 0.0014262, -0.0189003, -0.0038204, -0.0160218, 0.0203265
4: 0.0235673, 0.0504758, 0.0249442, 0.0463378, -0.0227705, 0.0255316

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.0310772, 0.1741992, 0.0312493, 0.1637233, -0.1326460, 0.1429499
1: -0.0160661, 0.0144748, -0.0154227, 0.0139402, -0.0300063, 0.0298974
2: 0.0269344, 0.0553142, 0.0272178, 0.0548256, -0.0278912, 0.0280964
3: -0.0198422, 0.0014262, -0.0196015, 0.0005169, -0.0203591, 0.0210277
4: 0.0235673, 0.0504758, 0.0238428, 0.0499271, -0.0263598, 0.0266331

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

## BFS NS instance: NS_A2_B2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0312119, 0.1642383, 0.0323709, 0.1233958, -0.0921839, 0.1318673
1: -0.0154898, 0.0140534, -0.0125881, 0.0111360, -0.0266258, 0.0266415
2: 0.0271566, 0.0549670, 0.0284928, 0.0515530, -0.0243964, 0.0264742
3: -0.0196514, 0.0005930, -0.0190458, -0.0031060, -0.0165454, 0.0196388
4: 0.0237919, 0.0500420, 0.0247660, 0.0465239, -0.0227320, 0.0252760

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A2_B2_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: 0.0310772, 0.1741992, 0.0323709, 0.1233958, -0.0923186, 0.1418282
1: -0.0160661, 0.0144748, -0.0125881, 0.0111360, -0.0272021, 0.0270629
2: 0.0269344, 0.0553142, 0.0284928, 0.0515530, -0.0246186, 0.0268214
3: -0.0198422, 0.0014262, -0.0190458, -0.0031060, -0.0167362, 0.0204720
4: 0.0235673, 0.0504758, 0.0247660, 0.0465239, -0.0229566, 0.0257099

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A2_B2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0312119, 0.1642383, 0.0310772, 0.1740944, -0.1428825, 0.1331610
1: -0.0154898, 0.0140534, -0.0160661, 0.0144619, -0.0299517, 0.0301194
2: 0.0271566, 0.0549670, 0.0269344, 0.0551495, -0.0279929, 0.0280326
3: -0.0196514, 0.0005930, -0.0198422, 0.0013388, -0.0209902, 0.0204352
4: 0.0237919, 0.0500420, 0.0235673, 0.0503064, -0.0265145, 0.0264747

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A2_B2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: 0.0310772, 0.1741992, 0.0310772, 0.1740944, -0.1430171, 0.1431220
1: -0.0160661, 0.0144748, -0.0160661, 0.0144619, -0.0305280, 0.0305408
2: 0.0269344, 0.0553142, 0.0269344, 0.0551495, -0.0282151, 0.0283798
3: -0.0198422, 0.0014262, -0.0198422, 0.0013388, -0.0211809, 0.0212684
4: 0.0235673, 0.0504758, 0.0235673, 0.0503064, -0.0267391, 0.0269086

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.51 + 231.03 = 233.54 seconds
