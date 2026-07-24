## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.088187946


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102)
1: (-0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898)
2: (-0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237)
3: (-0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035)
4: (-0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.73 + 0.84 = 1.58 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0899877, upper bound: 0.0899877

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895312, upper bound: 0.0896540
time: 0.23 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781
time: 0.23 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.53 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.53
Output dim: 0, lower bound: -0.0895312, upper bound: 0.0896540
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.53
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0303923, 0.0316036, -0.0390887, 0.0634215, -0.0938137, 0.0706923
1: -0.0419247, 0.0834490, -0.0557757, 0.1414140, -0.1833387, 0.1392248
2: -0.0824195, 0.1223470, -0.1134923, 0.1820314, -0.2644509, 0.2358393
3: -0.0466005, 0.0980623, -0.0636387, 0.1747649, -0.2213654, 0.1617010
4: -0.0908781, 0.1427428, -0.1363116, 0.2151742, -0.3060522, 0.2790544

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
time: 0.22 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
time: 0.23 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0358978, 0.0521367, -0.0390887, 0.0634215, -0.0993192, 0.0912254
1: -0.0510239, 0.1157805, -0.0557757, 0.1414140, -0.1924379, 0.1715562
2: -0.1029525, 0.1605961, -0.1134923, 0.1820314, -0.2849840, 0.2740884
3: -0.0576805, 0.1386453, -0.0636387, 0.1747649, -0.2324454, 0.2022839
4: -0.1189080, 0.1891569, -0.1363116, 0.2151742, -0.3340822, 0.3254685

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
time: 0.23 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781
time: 0.22 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.17 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1.17
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.17
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1.17
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1.17
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0303923, 0.0316036, -0.0303923, 0.0316036, -0.0619959, 0.0619959
1: -0.0419247, 0.0834490, -0.0419247, 0.0834490, -0.1253737, 0.1253737
2: -0.0824195, 0.1223470, -0.0824195, 0.1223470, -0.2047665, 0.2047665
3: -0.0466005, 0.0980623, -0.0466005, 0.0980623, -0.1446628, 0.1446628
4: -0.0908781, 0.1427428, -0.0908781, 0.1427428, -0.2336209, 0.2336209

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880314, upper bound: 0.0890560
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
time: 0.22 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0303923, 0.0316036, -0.0358978, 0.0521367, -0.0825290, 0.0675014
1: -0.0419247, 0.0834490, -0.0510239, 0.1157805, -0.1577052, 0.1344729
2: -0.0824195, 0.1223470, -0.1029525, 0.1605961, -0.2430156, 0.2252996
3: -0.0466005, 0.0980623, -0.0576805, 0.1386453, -0.1852457, 0.1557429
4: -0.0908781, 0.1427428, -0.1189080, 0.1891569, -0.2800349, 0.2616507

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880314, upper bound: 0.0890562
time: 0.23 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
time: 0.22 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0358978, 0.0521367, -0.0303923, 0.0316036, -0.0675014, 0.0825290
1: -0.0510239, 0.1157805, -0.0419247, 0.0834490, -0.1344729, 0.1577052
2: -0.1029525, 0.1605961, -0.0824195, 0.1223470, -0.2252996, 0.2430156
3: -0.0576805, 0.1386453, -0.0466005, 0.0980623, -0.1557429, 0.1852458
4: -0.1189080, 0.1891569, -0.0908781, 0.1427428, -0.2616507, 0.2800350

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0891589
time: 0.25 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
time: 0.24 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0358978, 0.0521367, -0.0358978, 0.0521367, -0.0880345, 0.0880345
1: -0.0510239, 0.1157805, -0.0510239, 0.1157805, -0.1668044, 0.1668043
2: -0.1029525, 0.1605961, -0.1029525, 0.1605961, -0.2635486, 0.2635487
3: -0.0576805, 0.1386453, -0.0576805, 0.1386453, -0.1963258, 0.1963258
4: -0.1189080, 0.1891569, -0.1189080, 0.1891569, -0.3080648, 0.3080649

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0892359
time: 0.25 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317
time: 0.24 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.22 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.22
Output dim: 0, lower bound: -0.0880314, upper bound: 0.0890560
NS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 1.22
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.22
Output dim: 0, lower bound: -0.0880314, upper bound: 0.0890562
NS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 1.22
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.22
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0891589
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 1.22
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.22
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0892359
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 1.22
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0285468, 0.0288364, -0.0303923, 0.0316036, -0.0601504, 0.0592287
1: -0.0394259, 0.0744049, -0.0419247, 0.0834490, -0.1228749, 0.1163296
2: -0.0761525, 0.1130898, -0.0824195, 0.1223470, -0.1984996, 0.1955093
3: -0.0434286, 0.0854480, -0.0466005, 0.0980623, -0.1414909, 0.1320485
4: -0.0817791, 0.1311970, -0.0908781, 0.1427428, -0.2245218, 0.2220750

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0285468, 0.0288364, -0.0358978, 0.0521367, -0.0806835, 0.0647341
1: -0.0394259, 0.0744049, -0.0510239, 0.1157805, -0.1552063, 0.1254288
2: -0.0761525, 0.1130898, -0.1029525, 0.1605961, -0.2367487, 0.2160423
3: -0.0434286, 0.0854480, -0.0576805, 0.1386453, -0.1820739, 0.1431285
4: -0.0817791, 0.1311970, -0.1189080, 0.1891569, -0.2709360, 0.2501049

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.25 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.24 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0318214, 0.0400522, -0.0303923, 0.0316036, -0.0634249, 0.0704445
1: -0.0462006, 0.0969325, -0.0419247, 0.0834490, -0.1296496, 0.1388572
2: -0.0906157, 0.1402107, -0.0824195, 0.1223470, -0.2129627, 0.2226302
3: -0.0531606, 0.1144974, -0.0466005, 0.0980623, -0.1512229, 0.1610979
4: -0.1036385, 0.1661279, -0.0908781, 0.1427428, -0.2463813, 0.2570060

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.24 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.24 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0318214, 0.0400522, -0.0358978, 0.0521367, -0.0839581, 0.0759500
1: -0.0462006, 0.0969325, -0.0510239, 0.1157805, -0.1619810, 0.1479563
2: -0.0906157, 0.1402107, -0.1029525, 0.1605961, -0.2512118, 0.2431632
3: -0.0531606, 0.1144974, -0.0576805, 0.1386453, -0.1918058, 0.1721780
4: -0.1036385, 0.1661279, -0.1189080, 0.1891569, -0.2927953, 0.2850358

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.27 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.26 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.25 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
NS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
NS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 1.25
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 1.58 + 12.49 = 14.07 seconds
