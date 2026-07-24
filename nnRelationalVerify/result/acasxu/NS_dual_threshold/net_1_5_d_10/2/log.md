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
execution time: IAR + RelationalAnalysis = 0.69 + 0.84 = 1.53 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0899877, upper bound: 0.0899877

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0897690, upper bound: 0.0898732
time: 0.21 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899608, upper bound: 0.0899608
time: 0.24 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.52 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.52
Output dim: 0, lower bound: -0.0897690, upper bound: 0.0898732
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.52
Output dim: 0, lower bound: -0.0899608, upper bound: 0.0899608

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0355980, 0.0491149, -0.0386665, 0.0612692, -0.0968673, 0.0877814
1: -0.0454062, 0.1038414, -0.0551382, 0.1370583, -0.1824645, 0.1589795
2: -0.1046092, 0.1549752, -0.1122055, 0.1787005, -0.2833098, 0.2671807
3: -0.0520488, 0.1242788, -0.0628254, 0.1687326, -0.2207814, 0.1871042
4: -0.1149560, 0.1775723, -0.1337108, 0.2112305, -0.3261864, 0.3112831

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896813, upper bound: 0.0896813
time: 0.21 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896813, upper bound: 0.0898732
time: 0.22 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0383316, 0.0603406, -0.0390887, 0.0634215, -0.1017531, 0.0994293
1: -0.0543237, 0.1360877, -0.0557757, 0.1414140, -0.1957377, 0.1918634
2: -0.1110429, 0.1767848, -0.1134923, 0.1820314, -0.2930743, 0.2902771
3: -0.0620208, 0.1679080, -0.0636387, 0.1747649, -0.2367857, 0.2315467
4: -0.1327446, 0.2087907, -0.1363116, 0.2151742, -0.3479187, 0.3451021

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888762, upper bound: 0.0897074
time: 0.22 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887913, upper bound: 0.0887913
time: 0.23 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.12 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1.12
Output dim: 0, lower bound: -0.0896813, upper bound: 0.0896813
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.12
Output dim: 0, lower bound: -0.0896813, upper bound: 0.0898732
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 1.12
Output dim: 0, lower bound: -0.0888762, upper bound: 0.0897074
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 1.12
Output dim: 0, lower bound: -0.0887913, upper bound: 0.0887913

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0355980, 0.0491149, -0.0355980, 0.0491149, -0.0847130, 0.0847130
1: -0.0454062, 0.1038414, -0.0454062, 0.1038414, -0.1492476, 0.1492476
2: -0.1046092, 0.1549752, -0.1046092, 0.1549752, -0.2595844, 0.2595844
3: -0.0520488, 0.1242788, -0.0520488, 0.1242788, -0.1763276, 0.1763276
4: -0.1149560, 0.1775723, -0.1149560, 0.1775723, -0.2925282, 0.2925282

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888446, upper bound: 0.0890278
time: 0.20 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886361, upper bound: 0.0886361
time: 0.21 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0355980, 0.0491149, -0.0383316, 0.0603406, -0.0959386, 0.0874465
1: -0.0454062, 0.1038414, -0.0543237, 0.1360877, -0.1814938, 0.1581650
2: -0.1046092, 0.1549752, -0.1110429, 0.1767848, -0.2813941, 0.2660180
3: -0.0520488, 0.1242788, -0.0620208, 0.1679080, -0.2199568, 0.1862996
4: -0.1149560, 0.1775723, -0.1327446, 0.2087907, -0.3237467, 0.3103168

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896273, upper bound: 0.0898715
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896813, upper bound: 0.0898717
time: 0.23 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -0.0358490, 0.0487337, -0.0390887, 0.0634215, -0.0992705, 0.0878224
1: -0.0507856, 0.1154432, -0.0557757, 0.1414140, -0.1921996, 0.1712189
2: -0.1024819, 0.1582004, -0.1134923, 0.1820314, -0.2845134, 0.2716927
3: -0.0580275, 0.1404007, -0.0636387, 0.1747649, -0.2327923, 0.2040394
4: -0.1195223, 0.1871233, -0.1363116, 0.2151742, -0.3346965, 0.3234349

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_A1_A1

### Relational analysis result of NS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888467, upper bound: 0.0896720
time: 0.23 seconds

## Relational analysis of NS_A2_A1_A2

### Relational analysis result of NS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888748, upper bound: 0.0897056
time: 0.23 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -0.0262249, 0.0476892, -0.0390887, 0.0634215, -0.0896463, 0.0867779
1: -0.0560731, 0.1461119, -0.0557757, 0.1414140, -0.1974871, 0.2018876
2: -0.0741902, 0.1451171, -0.1134923, 0.1820314, -0.2562216, 0.2586094
3: -0.0633892, 0.1894719, -0.0636387, 0.1747649, -0.2381541, 0.2531106
4: -0.1138687, 0.1800817, -0.1363116, 0.2151742, -0.3290429, 0.3163933

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879593, upper bound: 0.0879872
time: 0.23 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880320
time: 0.24 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.15 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.15
Output dim: 0, lower bound: -0.0888446, upper bound: 0.0890278
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.15
Output dim: 0, lower bound: -0.0886361, upper bound: 0.0886361
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.15
Output dim: 0, lower bound: -0.0896273, upper bound: 0.0898715
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.15
Output dim: 0, lower bound: -0.0896813, upper bound: 0.0898717
NS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 1.15
Output dim: 0, lower bound: -0.0888467, upper bound: 0.0896720
NS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 1.15
Output dim: 0, lower bound: -0.0888748, upper bound: 0.0897056
NS_A2_A2_B1, status: Status.VERIFIED, split count: 3, time: 1.15
Output dim: 0, lower bound: -0.0879593, upper bound: 0.0879872
NS_A2_A2_B2, status: Status.VERIFIED, split count: 3, time: 1.15
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880320

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0246740, 0.0221408, -0.0355980, 0.0491149, -0.0737889, 0.0577388
1: -0.0320872, 0.0524673, -0.0454062, 0.1038414, -0.1359286, 0.0978735
2: -0.0651265, 0.1000648, -0.1046092, 0.1549752, -0.2201016, 0.2046741
3: -0.0352156, 0.0550946, -0.0520488, 0.1242788, -0.1594944, 0.1071435
4: -0.0637753, 0.1115354, -0.1149560, 0.1775723, -0.2413475, 0.2264914

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887986, upper bound: 0.0887307
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886025, upper bound: 0.0886244
time: 0.22 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0342600, 0.0395509, -0.0355980, 0.0491149, -0.0833750, 0.0751490
1: -0.0425888, 0.0814996, -0.0454062, 0.1038414, -0.1464302, 0.1269058
2: -0.0995706, 0.1393745, -0.1046092, 0.1549752, -0.2545458, 0.2439837
3: -0.0484002, 0.0913292, -0.0520488, 0.1242788, -0.1726790, 0.1433780
4: -0.1022166, 0.1587072, -0.1149560, 0.1775723, -0.2797888, 0.2736630

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885881, upper bound: 0.0883389
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882910, upper bound: 0.0882910
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0289199, 0.0316199, -0.0383316, 0.0603406, -0.0892605, 0.0699515
1: -0.0364605, 0.0725038, -0.0543237, 0.1360877, -0.1725482, 0.1268275
2: -0.0836895, 0.1224094, -0.1110429, 0.1767848, -0.2604744, 0.2334523
3: -0.0432477, 0.0836505, -0.0620208, 0.1679080, -0.2111556, 0.1456713
4: -0.0884080, 0.1397506, -0.1327446, 0.2087907, -0.2971987, 0.2724952

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896826, upper bound: 0.0898522
time: 0.25 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896826, upper bound: 0.0898715
time: 0.26 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0341963, 0.0451829, -0.0383316, 0.0603406, -0.0945368, 0.0835145
1: -0.0427969, 0.0966445, -0.0543237, 0.1360877, -0.1788846, 0.1509682
2: -0.1001576, 0.1478489, -0.1110429, 0.1767848, -0.2769424, 0.2588918
3: -0.0493405, 0.1152938, -0.0620208, 0.1679080, -0.2172485, 0.1773146
4: -0.1093097, 0.1684246, -0.1327446, 0.2087907, -0.3181004, 0.3011690

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0897366, upper bound: 0.0898524
time: 0.23 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896811, upper bound: 0.0898717
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0281139, 0.0310250, -0.0390887, 0.0634215, -0.0915353, 0.0701137
1: -0.0406024, 0.0818455, -0.0557757, 0.1414140, -0.1820164, 0.1376212
2: -0.0773617, 0.1205507, -0.1134923, 0.1820314, -0.2593932, 0.2340430
3: -0.0474901, 0.0966178, -0.0636387, 0.1747649, -0.2222549, 0.1602564
4: -0.0884082, 0.1426494, -0.1363116, 0.2151742, -0.3035824, 0.2789610

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_A1_B1

### Relational analysis result of NS_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888467, upper bound: 0.0896720
time: 0.23 seconds

## Relational analysis of NS_A2_A1_A1_B2

### Relational analysis result of NS_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888467, upper bound: 0.0896720
time: 0.22 seconds

## BFS NS instance: NS_A2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0326413, 0.0406024, -0.0390887, 0.0634215, -0.0960627, 0.0796911
1: -0.0464028, 0.1029233, -0.0557757, 0.1414140, -0.1878169, 0.1586990
2: -0.0931736, 0.1439410, -0.1134923, 0.1820314, -0.2752050, 0.2574333
3: -0.0537068, 0.1251042, -0.0636387, 0.1747649, -0.2284718, 0.1887429
4: -0.1092366, 0.1700594, -0.1363116, 0.2151742, -0.3244108, 0.3063710

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_A2_B1

### Relational analysis result of NS_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888748, upper bound: 0.0897056
time: 0.25 seconds

## Relational analysis of NS_A2_A1_A2_B2

### Relational analysis result of NS_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888748, upper bound: 0.0897056
time: 0.25 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.19 seconds
NS_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 1.19
Output dim: 0, lower bound: -0.0887986, upper bound: 0.0887307
NS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 1.19
Output dim: 0, lower bound: -0.0886025, upper bound: 0.0886244
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 1.19
Output dim: 0, lower bound: -0.0885881, upper bound: 0.0883389
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 1.19
Output dim: 0, lower bound: -0.0882910, upper bound: 0.0882910
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.19
Output dim: 0, lower bound: -0.0896826, upper bound: 0.0898522
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.19
Output dim: 0, lower bound: -0.0896826, upper bound: 0.0898715
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.19
Output dim: 0, lower bound: -0.0897366, upper bound: 0.0898524
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.19
Output dim: 0, lower bound: -0.0896811, upper bound: 0.0898717
NS_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 1.19
Output dim: 0, lower bound: -0.0888467, upper bound: 0.0896720
NS_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 1.19
Output dim: 0, lower bound: -0.0888467, upper bound: 0.0896720
NS_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.19
Output dim: 0, lower bound: -0.0888748, upper bound: 0.0897056
NS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.19
Output dim: 0, lower bound: -0.0888748, upper bound: 0.0897056

## BFS NS instance: NS_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0240350, 0.0203458, -0.0355980, 0.0491149, -0.0731499, 0.0559438
1: -0.0301042, 0.0472260, -0.0454062, 0.1038414, -0.1339456, 0.0926322
2: -0.0630851, 0.0951283, -0.1046092, 0.1549752, -0.2180602, 0.1997375
3: -0.0331776, 0.0482606, -0.0520488, 0.1242788, -0.1574564, 0.1003094
4: -0.0595089, 0.1043710, -0.1149560, 0.1775723, -0.2370812, 0.2193269

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887986, upper bound: 0.0886988
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886025, upper bound: 0.0886244
time: 0.20 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2

### Relational analysis result of NS_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886025, upper bound: 0.0886244
time: 0.21 seconds

## BFS NS instance: NS_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0238656, 0.0210918, -0.0355980, 0.0491149, -0.0729805, 0.0566898
1: -0.0303834, 0.0492062, -0.0454062, 0.1038414, -0.1342248, 0.0946124
2: -0.0626324, 0.0968737, -0.1046092, 0.1549752, -0.2176076, 0.2014829
3: -0.0338869, 0.0508352, -0.0520488, 0.1242788, -0.1581657, 0.1028840
4: -0.0603797, 0.1074024, -0.1149560, 0.1775723, -0.2379519, 0.2223583

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886025, upper bound: 0.0886244
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886025, upper bound: 0.0886244
time: 0.21 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0303041, 0.0332229, -0.0355980, 0.0491149, -0.0794190, 0.0688209
1: -0.0381444, 0.0714089, -0.0454062, 0.1038414, -0.1419857, 0.1168151
2: -0.0873048, 0.1254651, -0.1046092, 0.1549752, -0.2422799, 0.2300744
3: -0.0446786, 0.0786664, -0.0520488, 0.1242788, -0.1689574, 0.1307152
4: -0.0888710, 0.1432975, -0.1149560, 0.1775723, -0.2664433, 0.2582535

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884308, upper bound: 0.0883389
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884308, upper bound: 0.0883389
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0335580, 0.0372201, -0.0355980, 0.0491149, -0.0826729, 0.0728182
1: -0.0408165, 0.0770043, -0.0454062, 0.1038414, -0.1446578, 0.1224105
2: -0.0973802, 0.1353522, -0.1046092, 0.1549752, -0.2523554, 0.2399613
3: -0.0471661, 0.0854527, -0.0520488, 0.1242788, -0.1714449, 0.1375015
4: -0.0990589, 0.1538413, -0.1149560, 0.1775723, -0.2766311, 0.2687973

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881337, upper bound: 0.0882910
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881337, upper bound: 0.0882910
time: 0.22 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0289199, 0.0316199, -0.0303880, 0.0381159, -0.0670358, 0.0620079
1: -0.0364605, 0.0725038, -0.0436567, 0.0986741, -0.1351346, 0.1161605
2: -0.0836895, 0.1224094, -0.0858032, 0.1358688, -0.2195583, 0.2082126
3: -0.0432477, 0.0836505, -0.0513058, 0.1197162, -0.1629638, 0.1349563
4: -0.0884080, 0.1397506, -0.1010720, 0.1610686, -0.2494766, 0.2408226

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894839, upper bound: 0.0897163
time: 0.25 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894730, upper bound: 0.0897311
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0289199, 0.0316199, -0.0352700, 0.0520305, -0.0809503, 0.0668899
1: -0.0364605, 0.0725038, -0.0497746, 0.1236877, -0.1601482, 0.1222784
2: -0.0836895, 0.1224094, -0.1017607, 0.1623120, -0.2460016, 0.2241702
3: -0.0432477, 0.0836505, -0.0576897, 0.1530293, -0.1962769, 0.1413402
4: -0.0884080, 0.1397506, -0.1224101, 0.1915598, -0.2799676, 0.2621607

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894839, upper bound: 0.0897506
time: 0.22 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894730, upper bound: 0.0897497
time: 0.23 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0341963, 0.0451829, -0.0303880, 0.0381159, -0.0723121, 0.0755709
1: -0.0427969, 0.0966445, -0.0436567, 0.0986741, -0.1414710, 0.1403012
2: -0.1001576, 0.1478489, -0.0858032, 0.1358688, -0.2360264, 0.2336521
3: -0.0493405, 0.1152938, -0.0513058, 0.1197162, -0.1690567, 0.1665996
4: -0.1093097, 0.1684246, -0.1010720, 0.1610686, -0.2703783, 0.2694965

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890695, upper bound: 0.0892715
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891090, upper bound: 0.0890427
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0341963, 0.0451829, -0.0352700, 0.0520305, -0.0862268, 0.0804528
1: -0.0427969, 0.0966445, -0.0497746, 0.1236877, -0.1664846, 0.1464191
2: -0.1001576, 0.1478489, -0.1017607, 0.1623120, -0.2624696, 0.2496097
3: -0.0493405, 0.1152938, -0.0576897, 0.1530293, -0.2023698, 0.1729835
4: -0.1093097, 0.1684246, -0.1224101, 0.1915598, -0.3008694, 0.2908345

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890695, upper bound: 0.0892734
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891090, upper bound: 0.0890427
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0281139, 0.0310250, -0.0366452, 0.0517803, -0.0798942, 0.0676701
1: -0.0406024, 0.0818455, -0.0522854, 0.1216848, -0.1622871, 0.1341309
2: -0.0773617, 0.1205507, -0.1050165, 0.1634838, -0.2408455, 0.2255672
3: -0.0474901, 0.0966178, -0.0597044, 0.1486967, -0.1961867, 0.1563222
4: -0.0884082, 0.1426494, -0.1233429, 0.1935518, -0.2819600, 0.2659923

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A1_B1_A1

### Relational analysis result of NS_A2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888426, upper bound: 0.0896720
time: 0.23 seconds

## Relational analysis of NS_A2_A1_A1_B1_A2

### Relational analysis result of NS_A2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888467, upper bound: 0.0896720
time: 0.25 seconds

## BFS NS instance: NS_A2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0281139, 0.0310250, -0.0284880, 0.0553090, -0.0834228, 0.0595129
1: -0.0406024, 0.0818455, -0.0604131, 0.1616067, -0.2022091, 0.1422586
2: -0.0773617, 0.1205507, -0.0813747, 0.1602389, -0.2376007, 0.2019254
3: -0.0474901, 0.0966178, -0.0681267, 0.2105855, -0.2580756, 0.1647445
4: -0.0884082, 0.1426494, -0.1251932, 0.1995282, -0.2879363, 0.2678426

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A1_B2_B1

### Relational analysis result of NS_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888353, upper bound: 0.0896673
time: 0.22 seconds

## Relational analysis of NS_A2_A1_A1_B2_B2

### Relational analysis result of NS_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888467, upper bound: 0.0896720
time: 0.24 seconds

## BFS NS instance: NS_A2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0326413, 0.0406024, -0.0366452, 0.0517803, -0.0844216, 0.0772475
1: -0.0464028, 0.1029233, -0.0522854, 0.1216848, -0.1680876, 0.1552087
2: -0.0931736, 0.1439410, -0.1050165, 0.1634838, -0.2566574, 0.2489575
3: -0.0537068, 0.1251042, -0.0597044, 0.1486967, -0.2024035, 0.1848086
4: -0.1092366, 0.1700594, -0.1233429, 0.1935518, -0.3027885, 0.2934023

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_A1_A2_B1_B1

### Relational analysis result of NS_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882623, upper bound: 0.0894891
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A2_B1_B2

### Relational analysis result of NS_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882623, upper bound: 0.0896398
time: 0.24 seconds

## BFS NS instance: NS_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0326413, 0.0406024, -0.0284880, 0.0553090, -0.0879502, 0.0690903
1: -0.0464028, 0.1029233, -0.0604131, 0.1616067, -0.2080095, 0.1633364
2: -0.0931736, 0.1439410, -0.0813747, 0.1602389, -0.2534125, 0.2253157
3: -0.0537068, 0.1251042, -0.0681267, 0.2105855, -0.2642924, 0.1932309
4: -0.1092366, 0.1700594, -0.1251932, 0.1995282, -0.3087648, 0.2952526

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A2_B2_B1

### Relational analysis result of NS_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888634, upper bound: 0.0897000
time: 0.26 seconds

## Relational analysis of NS_A2_A1_A2_B2_B2

### Relational analysis result of NS_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888748, upper bound: 0.0897044
time: 0.24 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 1.22 seconds
NS_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0886025, upper bound: 0.0886244
NS_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0886025, upper bound: 0.0886244
NS_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0886025, upper bound: 0.0886244
NS_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0886025, upper bound: 0.0886244
NS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0884308, upper bound: 0.0883389
NS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0884308, upper bound: 0.0883389
NS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0881337, upper bound: 0.0882910
NS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0881337, upper bound: 0.0882910
NS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0894839, upper bound: 0.0897163
NS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0894730, upper bound: 0.0897311
NS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0894839, upper bound: 0.0897506
NS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0894730, upper bound: 0.0897497
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0890695, upper bound: 0.0892715
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0891090, upper bound: 0.0890427
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0890695, upper bound: 0.0892734
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0891090, upper bound: 0.0890427
NS_A2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0888426, upper bound: 0.0896720
NS_A2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0888467, upper bound: 0.0896720
NS_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0888353, upper bound: 0.0896673
NS_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0888467, upper bound: 0.0896720
NS_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0882623, upper bound: 0.0894891
NS_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0882623, upper bound: 0.0896398
NS_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0888634, upper bound: 0.0897000
NS_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 1.22
Output dim: 0, lower bound: -0.0888748, upper bound: 0.0897044

## BFS NS instance: NS_A1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0240350, 0.0203458, -0.0336301, 0.0464292, -0.0704642, 0.0539759
1: -0.0301042, 0.0472260, -0.0425000, 0.0980784, -0.1281826, 0.0897260
2: -0.0630851, 0.0951283, -0.0976411, 0.1494459, -0.2125309, 0.1927693
3: -0.0331776, 0.0482606, -0.0496717, 0.1168486, -0.1500262, 0.0979323
4: -0.0595089, 0.1043710, -0.1071893, 0.1712112, -0.2307201, 0.2115602

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886413, upper bound: 0.0887307
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886413, upper bound: 0.0887307
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0240350, 0.0203458, -0.0346241, 0.0461729, -0.0702079, 0.0549699
1: -0.0301042, 0.0472260, -0.0430924, 0.0979473, -0.1280515, 0.0903184
2: -0.0630851, 0.0951283, -0.1013632, 0.1495815, -0.2126665, 0.1964914
3: -0.0331776, 0.0482606, -0.0501156, 0.1165557, -0.1497332, 0.0983762
4: -0.0595089, 0.1043710, -0.1102601, 0.1708822, -0.2303911, 0.2146311

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886413, upper bound: 0.0887307
time: 0.24 seconds

## Relational analysis of NS_A1_B1_A1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886413, upper bound: 0.0887307
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0238656, 0.0210918, -0.0336301, 0.0464292, -0.0702948, 0.0547219
1: -0.0303834, 0.0492062, -0.0425000, 0.0980784, -0.1284618, 0.0917062
2: -0.0626324, 0.0968737, -0.0976411, 0.1494459, -0.2120783, 0.1945147
3: -0.0338869, 0.0508352, -0.0496717, 0.1168486, -0.1507355, 0.1005069
4: -0.0603797, 0.1074024, -0.1071893, 0.1712112, -0.2315909, 0.2145916

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884452, upper bound: 0.0886244
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884452, upper bound: 0.0886244
time: 0.21 seconds

## BFS NS instance: NS_A1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0238656, 0.0210918, -0.0346241, 0.0461729, -0.0700385, 0.0557159
1: -0.0303834, 0.0492062, -0.0430924, 0.0979473, -0.1283307, 0.0922986
2: -0.0626324, 0.0968737, -0.1013632, 0.1495815, -0.2122139, 0.1982368
3: -0.0338869, 0.0508352, -0.0501156, 0.1165557, -0.1504426, 0.1009508
4: -0.0603797, 0.1074024, -0.1102601, 0.1708822, -0.2312618, 0.2176625

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884452, upper bound: 0.0886244
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884452, upper bound: 0.0886244
time: 0.21 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0303041, 0.0332229, -0.0246740, 0.0221408, -0.0524449, 0.0578968
1: -0.0381444, 0.0714089, -0.0320872, 0.0524673, -0.0906117, 0.1034961
2: -0.0873048, 0.1254651, -0.0651265, 0.1000648, -0.1873696, 0.1905916
3: -0.0446786, 0.0786664, -0.0352156, 0.0550946, -0.0997732, 0.1138819
4: -0.0888710, 0.1432975, -0.0637753, 0.1115354, -0.2004064, 0.2070728

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_A1_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884308, upper bound: 0.0883389
time: 0.22 seconds

## Relational analysis of NS_A1_B1_A2_A1_B1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884308, upper bound: 0.0883389
time: 0.25 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0303041, 0.0332229, -0.0342600, 0.0395509, -0.0698550, 0.0674829
1: -0.0381444, 0.0714089, -0.0425888, 0.0814996, -0.1196440, 0.1139977
2: -0.0873048, 0.1254651, -0.0995706, 0.1393745, -0.2266793, 0.2250358
3: -0.0446786, 0.0786664, -0.0484002, 0.0913292, -0.1360078, 0.1270666
4: -0.0888710, 0.1432975, -0.1022166, 0.1587072, -0.2475781, 0.2455141

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884308, upper bound: 0.0883067
time: 0.21 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884308, upper bound: 0.0883389
time: 0.25 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884308, upper bound: 0.0883389
time: 0.23 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0335580, 0.0372201, -0.0246740, 0.0221408, -0.0556988, 0.0618941
1: -0.0408165, 0.0770043, -0.0320872, 0.0524673, -0.0932838, 0.1090915
2: -0.0973802, 0.1353522, -0.0651265, 0.1000648, -0.1974451, 0.2004786
3: -0.0471661, 0.0854527, -0.0352156, 0.0550946, -0.1022608, 0.1206683
4: -0.0990589, 0.1538413, -0.0637753, 0.1115354, -0.2105943, 0.2176166

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881337, upper bound: 0.0882910
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881337, upper bound: 0.0882910
time: 0.24 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0335580, 0.0372201, -0.0342600, 0.0395509, -0.0731089, 0.0714801
1: -0.0408165, 0.0770043, -0.0425888, 0.0814996, -0.1223161, 0.1195931
2: -0.0973802, 0.1353522, -0.0995706, 0.1393745, -0.2367548, 0.2349228
3: -0.0471661, 0.0854527, -0.0484002, 0.0913292, -0.1384953, 0.1338530
4: -0.0990589, 0.1538413, -0.1022166, 0.1587072, -0.2577661, 0.2560579

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881337, upper bound: 0.0882910
time: 0.23 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881337, upper bound: 0.0882910
time: 0.23 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0289199, 0.0316199, -0.0245046, 0.0304916, -0.0594115, 0.0561245
1: -0.0364605, 0.0725038, -0.0370066, 0.0861242, -0.1225847, 0.1095104
2: -0.0836895, 0.1224094, -0.0676304, 0.1160857, -0.1997753, 0.1900398
3: -0.0432477, 0.0836505, -0.0456015, 0.1057874, -0.1490351, 0.1292520
4: -0.0884080, 0.1397506, -0.0840454, 0.1383576, -0.2267656, 0.2237960

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894806, upper bound: 0.0897163
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894806, upper bound: 0.0897163
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0289199, 0.0316199, -0.0293784, 0.0348040, -0.0637239, 0.0609982
1: -0.0364605, 0.0725038, -0.0411437, 0.0922270, -0.1286876, 0.1136475
2: -0.0836895, 0.1224094, -0.0824305, 0.1296569, -0.2133465, 0.2048399
3: -0.0432477, 0.0836505, -0.0493116, 0.1114420, -0.1546896, 0.1329620
4: -0.0884080, 0.1397506, -0.0963648, 0.1533624, -0.2417704, 0.2361154

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886940, upper bound: 0.0891682
time: 0.25 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887542, upper bound: 0.0889951
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0289199, 0.0316199, -0.0298293, 0.0411537, -0.0700736, 0.0614492
1: -0.0364605, 0.0725038, -0.0437109, 0.1075221, -0.1439826, 0.1162146
2: -0.0836895, 0.1224094, -0.0845616, 0.1400728, -0.2237623, 0.2069710
3: -0.0432477, 0.0836505, -0.0523840, 0.1329535, -0.1762011, 0.1360344
4: -0.0884080, 0.1397506, -0.1043052, 0.1662240, -0.2546320, 0.2440558

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886121, upper bound: 0.0891797
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887520, upper bound: 0.0890131
time: 0.20 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0289199, 0.0316199, -0.0339535, 0.0482027, -0.0771226, 0.0655734
1: -0.0364605, 0.0725038, -0.0470253, 0.1167137, -0.1531743, 0.1195291
2: -0.0836895, 0.1224094, -0.0978653, 0.1553957, -0.2390852, 0.2202748
3: -0.0432477, 0.0836505, -0.0554273, 0.1441097, -0.1873574, 0.1390777
4: -0.0884080, 0.1397506, -0.1172390, 0.1831006, -0.2715085, 0.2569896

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885969, upper bound: 0.0891538
time: 0.23 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887073, upper bound: 0.0890131
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0237667, 0.0206726, -0.0303880, 0.0381159, -0.0618826, 0.0510606
1: -0.0301767, 0.0481623, -0.0436567, 0.0986741, -0.1288508, 0.0918190
2: -0.0622640, 0.0955307, -0.0858032, 0.1358688, -0.1981328, 0.1813339
3: -0.0332325, 0.0495364, -0.0513058, 0.1197162, -0.1529487, 0.1008422
4: -0.0597730, 0.1052151, -0.1010720, 0.1610686, -0.2208416, 0.2062870

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886306, upper bound: 0.0888775
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886306, upper bound: 0.0889019
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0322836, 0.0334207, -0.0303880, 0.0381159, -0.0703995, 0.0638087
1: -0.0396060, 0.0710703, -0.0436567, 0.0986741, -0.1382801, 0.1147270
2: -0.0933639, 0.1287424, -0.0858032, 0.1358688, -0.2292327, 0.2145456
3: -0.0451630, 0.0779622, -0.0513058, 0.1197162, -0.1648792, 0.1292680
4: -0.0942796, 0.1455511, -0.1010720, 0.1610686, -0.2553482, 0.2466230

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886306, upper bound: 0.0890183
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886306, upper bound: 0.0890427
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0237667, 0.0206726, -0.0352700, 0.0520305, -0.0757972, 0.0559426
1: -0.0301767, 0.0481623, -0.0497746, 0.1236877, -0.1538644, 0.0979369
2: -0.0622640, 0.0955307, -0.1017607, 0.1623120, -0.2245761, 0.1972914
3: -0.0332325, 0.0495364, -0.0576897, 0.1530293, -0.1862619, 0.1072261
4: -0.0597730, 0.1052151, -0.1224101, 0.1915598, -0.2513328, 0.2276252

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886439, upper bound: 0.0888775
time: 0.27 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886439, upper bound: 0.0889019
time: 0.22 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0322836, 0.0334207, -0.0352700, 0.0520305, -0.0843140, 0.0686906
1: -0.0396060, 0.0710703, -0.0497746, 0.1236877, -0.1632937, 0.1208448
2: -0.0933639, 0.1287424, -0.1017607, 0.1623120, -0.2556759, 0.2305031
3: -0.0451630, 0.0779622, -0.0576897, 0.1530293, -0.1981922, 0.1356519
4: -0.0942796, 0.1455511, -0.1224101, 0.1915598, -0.2858393, 0.2679612

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886439, upper bound: 0.0890183
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886439, upper bound: 0.0890427
time: 0.23 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0215165, 0.0264666, -0.0366452, 0.0517803, -0.0732968, 0.0631117
1: -0.0332614, 0.0721512, -0.0522854, 0.1216848, -0.1549461, 0.1244366
2: -0.0573547, 0.1032774, -0.1050165, 0.1634838, -0.2208384, 0.2082939
3: -0.0413794, 0.0860633, -0.0597044, 0.1486967, -0.1900760, 0.1457677
4: -0.0701147, 0.1229001, -0.1233429, 0.1935518, -0.2636665, 0.2462430

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_A1_A1_B1_A1_B1

### Relational analysis result of NS_A2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896052, upper bound: 0.0898560
time: 0.23 seconds

## Relational analysis of NS_A2_A1_A1_B1_A1_B2

### Relational analysis result of NS_A2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896052, upper bound: 0.0898724
time: 0.24 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0271535, 0.0296738, -0.0366452, 0.0517803, -0.0789338, 0.0663190
1: -0.0381406, 0.0776904, -0.0522854, 0.1216848, -0.1598253, 0.1299758
2: -0.0742795, 0.1164777, -0.1050165, 0.1634838, -0.2377633, 0.2214942
3: -0.0456244, 0.0911491, -0.0597044, 0.1486967, -0.1943210, 0.1508535
4: -0.0842188, 0.1373387, -0.1233429, 0.1935518, -0.2777707, 0.2606816

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_A1_A1_B1_A2_B1

### Relational analysis result of NS_A2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898335, upper bound: 0.0898520
time: 0.25 seconds

## Relational analysis of NS_A2_A1_A1_B1_A2_B2

### Relational analysis result of NS_A2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898335, upper bound: 0.0898683
time: 0.25 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0281139, 0.0310250, -0.0264032, 0.0510042, -0.0791181, 0.0574282
1: -0.0406024, 0.0818455, -0.0573585, 0.1537825, -0.1943849, 0.1392040
2: -0.0773617, 0.1205507, -0.0757487, 0.1525275, -0.2298892, 0.1962994
3: -0.0474901, 0.0966178, -0.0655027, 0.2003427, -0.2478328, 0.1621204
4: -0.0884082, 0.1426494, -0.1182454, 0.1901293, -0.2785375, 0.2608948

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A1_B2_B1_A1

### Relational analysis result of NS_A2_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881425, upper bound: 0.0889542
time: 0.23 seconds

## Relational analysis of NS_A2_A1_A1_B2_B1_A2

### Relational analysis result of NS_A2_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881460, upper bound: 0.0892236
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0281139, 0.0310250, -0.0275337, 0.0522360, -0.0803499, 0.0585587
1: -0.0406024, 0.0818455, -0.0577662, 0.1554659, -0.1960683, 0.1396117
2: -0.0773617, 0.1205507, -0.0781244, 0.1540693, -0.2314311, 0.1986751
3: -0.0474901, 0.0966178, -0.0657998, 0.2024251, -0.2499152, 0.1624175
4: -0.0884082, 0.1426494, -0.1204030, 0.1915261, -0.2799342, 0.2630524

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A1_B2_B2_A1

### Relational analysis result of NS_A2_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881425, upper bound: 0.0889656
time: 0.23 seconds

## Relational analysis of NS_A2_A1_A1_B2_B2_A2

### Relational analysis result of NS_A2_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881460, upper bound: 0.0892236
time: 0.24 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0326413, 0.0406024, -0.0288182, 0.0320959, -0.0647372, 0.0694205
1: -0.0464028, 0.1029233, -0.0419773, 0.0852667, -0.1316696, 0.1449006
2: -0.0931736, 0.1439410, -0.0796866, 0.1238025, -0.2169761, 0.2236276
3: -0.0537068, 0.1251042, -0.0488805, 0.1012414, -0.1549482, 0.1739847
4: -0.1092366, 0.1700594, -0.0916779, 0.1468575, -0.2560941, 0.2617373

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A2_B1_B1_B1

### Relational analysis result of NS_A2_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898724, upper bound: 0.0898256
time: 0.26 seconds

## Relational analysis of NS_A2_A1_A2_B1_B1_B2

### Relational analysis result of NS_A2_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898499, upper bound: 0.0898335
time: 0.28 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0326413, 0.0406024, -0.0335036, 0.0434306, -0.0760719, 0.0741060
1: -0.0464028, 0.1029233, -0.0478994, 0.1090685, -0.1554714, 0.1508226
2: -0.0931736, 0.1439410, -0.0958872, 0.1491258, -0.2422994, 0.2398282
3: -0.0537068, 0.1251042, -0.0553259, 0.1333559, -0.1870627, 0.1804301
4: -0.1092366, 0.1700594, -0.1132077, 0.1763901, -0.2856267, 0.2832671

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A2_B1_B2_A1

### Relational analysis result of NS_A2_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898518, upper bound: 0.0898991
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A2_B1_B2_A2

### Relational analysis result of NS_A2_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898499, upper bound: 0.0898335
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0326413, 0.0406024, -0.0264032, 0.0510042, -0.0836455, 0.0670056
1: -0.0464028, 0.1029233, -0.0573585, 0.1537825, -0.2001853, 0.1602818
2: -0.0931736, 0.1439410, -0.0757487, 0.1525275, -0.2457010, 0.2196897
3: -0.0537068, 0.1251042, -0.0655027, 0.2003427, -0.2540495, 0.1906068
4: -0.1092366, 0.1700594, -0.1182454, 0.1901293, -0.2993660, 0.2883048

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A2_B2_B1_A1

### Relational analysis result of NS_A2_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881544, upper bound: 0.0889778
time: 0.23 seconds

## Relational analysis of NS_A2_A1_A2_B2_B1_A2

### Relational analysis result of NS_A2_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881638, upper bound: 0.0892348
time: 0.22 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0326413, 0.0406024, -0.0275337, 0.0522360, -0.0848773, 0.0681361
1: -0.0464028, 0.1029233, -0.0577662, 0.1554659, -0.2018687, 0.1606894
2: -0.0931736, 0.1439410, -0.0781244, 0.1540693, -0.2472429, 0.2220653
3: -0.0537068, 0.1251042, -0.0657998, 0.2024251, -0.2561319, 0.1909040
4: -0.1092366, 0.1700594, -0.1204030, 0.1915261, -0.3007627, 0.2904625

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A2_B2_B2_A1

### Relational analysis result of NS_A2_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881544, upper bound: 0.0889911
time: 0.25 seconds

## Relational analysis of NS_A2_A1_A2_B2_B2_A2

### Relational analysis result of NS_A2_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881638, upper bound: 0.0892348
time: 0.26 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 1.26 seconds
NS_A1_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0886413, upper bound: 0.0887307
NS_A1_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0886413, upper bound: 0.0887307
NS_A1_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0886413, upper bound: 0.0887307
NS_A1_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0886413, upper bound: 0.0887307
NS_A1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0884452, upper bound: 0.0886244
NS_A1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0884452, upper bound: 0.0886244
NS_A1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0884452, upper bound: 0.0886244
NS_A1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0884452, upper bound: 0.0886244
NS_A1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0884308, upper bound: 0.0883389
NS_A1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0884308, upper bound: 0.0883389
NS_A1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0884308, upper bound: 0.0883389
NS_A1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0884308, upper bound: 0.0883389
NS_A1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0881337, upper bound: 0.0882910
NS_A1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0881337, upper bound: 0.0882910
NS_A1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0881337, upper bound: 0.0882910
NS_A1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0881337, upper bound: 0.0882910
NS_A1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0894806, upper bound: 0.0897163
NS_A1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0894806, upper bound: 0.0897163
NS_A1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0886940, upper bound: 0.0891682
NS_A1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0887542, upper bound: 0.0889951
NS_A1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0886121, upper bound: 0.0891797
NS_A1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0887520, upper bound: 0.0890131
NS_A1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0885969, upper bound: 0.0891538
NS_A1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0887073, upper bound: 0.0890131
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0886306, upper bound: 0.0888775
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0886306, upper bound: 0.0889019
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0886306, upper bound: 0.0890183
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0886306, upper bound: 0.0890427
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0886439, upper bound: 0.0888775
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0886439, upper bound: 0.0889019
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0886439, upper bound: 0.0890183
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0886439, upper bound: 0.0890427
NS_A2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0896052, upper bound: 0.0898560
NS_A2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0896052, upper bound: 0.0898724
NS_A2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0898335, upper bound: 0.0898520
NS_A2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0898335, upper bound: 0.0898683
NS_A2_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0881425, upper bound: 0.0889542
NS_A2_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0881460, upper bound: 0.0892236
NS_A2_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0881425, upper bound: 0.0889656
NS_A2_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0881460, upper bound: 0.0892236
NS_A2_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0898724, upper bound: 0.0898256
NS_A2_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0898499, upper bound: 0.0898335
NS_A2_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0898518, upper bound: 0.0898991
NS_A2_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0898499, upper bound: 0.0898335
NS_A2_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0881544, upper bound: 0.0889778
NS_A2_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0881638, upper bound: 0.0892348
NS_A2_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0881544, upper bound: 0.0889911
NS_A2_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 1.26
Output dim: 0, lower bound: -0.0881638, upper bound: 0.0892348

## BFS NS instance: NS_A1_B1_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0240350, 0.0203458, -0.0240350, 0.0203458, -0.0443808, 0.0443808
1: -0.0301042, 0.0472260, -0.0301042, 0.0472260, -0.0773302, 0.0773302
2: -0.0630851, 0.0951283, -0.0630851, 0.0951283, -0.1582133, 0.1582133
3: -0.0331776, 0.0482606, -0.0331776, 0.0482606, -0.0814381, 0.0814381
4: -0.0595089, 0.1043710, -0.0595089, 0.1043710, -0.1638799, 0.1638799

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

## BFS NS instance: NS_A1_B1_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0240350, 0.0203458, -0.0303041, 0.0332229, -0.0572578, 0.0506499
1: -0.0301042, 0.0472260, -0.0381444, 0.0714089, -0.1015131, 0.0853704
2: -0.0630851, 0.0951283, -0.0873048, 0.1254651, -0.1885502, 0.1824330
3: -0.0331776, 0.0482606, -0.0446786, 0.0786664, -0.1118439, 0.0929392
4: -0.0595089, 0.1043710, -0.0888710, 0.1432975, -0.2028064, 0.1932420

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

## BFS NS instance: NS_A1_B1_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0240350, 0.0203458, -0.0238656, 0.0210918, -0.0451267, 0.0442114
1: -0.0301042, 0.0472260, -0.0303834, 0.0492062, -0.0793105, 0.0776094
2: -0.0630851, 0.0951283, -0.0626324, 0.0968737, -0.1599587, 0.1577607
3: -0.0331776, 0.0482606, -0.0338869, 0.0508352, -0.0840128, 0.0821475
4: -0.0595089, 0.1043710, -0.0603797, 0.1074024, -0.1669113, 0.1647507

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

## BFS NS instance: NS_A1_B1_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0240350, 0.0203458, -0.0335580, 0.0372201, -0.0612551, 0.0539038
1: -0.0301042, 0.0472260, -0.0408165, 0.0770043, -0.1071085, 0.0880425
2: -0.0630851, 0.0951283, -0.0973802, 0.1353522, -0.1984371, 0.1925085
3: -0.0331776, 0.0482606, -0.0471661, 0.0854527, -0.1186303, 0.0954267
4: -0.0595089, 0.1043710, -0.0990589, 0.1538413, -0.2133503, 0.2034298

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

## BFS NS instance: NS_A1_B1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0238656, 0.0210918, -0.0240350, 0.0203458, -0.0442114, 0.0451267
1: -0.0303834, 0.0492062, -0.0301042, 0.0472260, -0.0776094, 0.0793105
2: -0.0626324, 0.0968737, -0.0630851, 0.0951283, -0.1577607, 0.1599587
3: -0.0338869, 0.0508352, -0.0331776, 0.0482606, -0.0821475, 0.0840128
4: -0.0603797, 0.1074024, -0.0595089, 0.1043710, -0.1647507, 0.1669113

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

## BFS NS instance: NS_A1_B1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0238656, 0.0210918, -0.0303041, 0.0332229, -0.0570884, 0.0513959
1: -0.0303834, 0.0492062, -0.0381444, 0.0714089, -0.1017923, 0.0873506
2: -0.0626324, 0.0968737, -0.0873048, 0.1254651, -0.1880975, 0.1841784
3: -0.0338869, 0.0508352, -0.0446786, 0.0786664, -0.1125533, 0.0955138
4: -0.0603797, 0.1074024, -0.0888710, 0.1432975, -0.2036772, 0.1962733

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

## BFS NS instance: NS_A1_B1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0238656, 0.0210918, -0.0238656, 0.0210918, -0.0449573, 0.0449573
1: -0.0303834, 0.0492062, -0.0303834, 0.0492062, -0.0795897, 0.0795897
2: -0.0626324, 0.0968737, -0.0626324, 0.0968737, -0.1595061, 0.1595061
3: -0.0338869, 0.0508352, -0.0338869, 0.0508352, -0.0847222, 0.0847222
4: -0.0603797, 0.1074024, -0.0603797, 0.1074024, -0.1677820, 0.1677820

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

## BFS NS instance: NS_A1_B1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0238656, 0.0210918, -0.0335580, 0.0372201, -0.0610857, 0.0546498
1: -0.0303834, 0.0492062, -0.0408165, 0.0770043, -0.1073877, 0.0900227
2: -0.0626324, 0.0968737, -0.0973802, 0.1353522, -0.1979846, 0.1942539
3: -0.0338869, 0.0508352, -0.0471661, 0.0854527, -0.1193396, 0.0980013
4: -0.0603797, 0.1074024, -0.0990589, 0.1538413, -0.2142210, 0.2064612

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

## BFS NS instance: NS_A1_B1_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0303041, 0.0332229, -0.0240350, 0.0203458, -0.0506499, 0.0572578
1: -0.0381444, 0.0714089, -0.0301042, 0.0472260, -0.0853704, 0.1015131
2: -0.0873048, 0.1254651, -0.0630851, 0.0951283, -0.1824330, 0.1885502
3: -0.0446786, 0.0786664, -0.0331776, 0.0482606, -0.0929392, 0.1118439
4: -0.0888710, 0.1432975, -0.0595089, 0.1043710, -0.1932420, 0.2028064

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## BFS NS instance: NS_A1_B1_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0303041, 0.0332229, -0.0238656, 0.0210918, -0.0513959, 0.0570884
1: -0.0381444, 0.0714089, -0.0303834, 0.0492062, -0.0873506, 0.1017923
2: -0.0873048, 0.1254651, -0.0626324, 0.0968737, -0.1841784, 0.1880975
3: -0.0446786, 0.0786664, -0.0338869, 0.0508352, -0.0955138, 0.1125533
4: -0.0888710, 0.1432975, -0.0603797, 0.1074024, -0.1962733, 0.2036772

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## BFS NS instance: NS_A1_B1_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0303041, 0.0332229, -0.0303041, 0.0332229, -0.0635270, 0.0635270
1: -0.0381444, 0.0714089, -0.0381444, 0.0714089, -0.1095532, 0.1095532
2: -0.0873048, 0.1254651, -0.0873048, 0.1254651, -0.2127699, 0.2127699
3: -0.0446786, 0.0786664, -0.0446786, 0.0786664, -0.1233450, 0.1233450
4: -0.0888710, 0.1432975, -0.0888710, 0.1432975, -0.2321685, 0.2321685

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

## BFS NS instance: NS_A1_B1_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0303041, 0.0332229, -0.0335580, 0.0372201, -0.0675242, 0.0667809
1: -0.0381444, 0.0714089, -0.0408165, 0.0770043, -0.1151486, 0.1122254
2: -0.0873048, 0.1254651, -0.0973802, 0.1353522, -0.2226569, 0.2228454
3: -0.0446786, 0.0786664, -0.0471661, 0.0854527, -0.1301313, 0.1258325
4: -0.0888710, 0.1432975, -0.0990589, 0.1538413, -0.2427123, 0.2423564

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

## BFS NS instance: NS_A1_B1_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0335580, 0.0372201, -0.0240350, 0.0203458, -0.0539038, 0.0612551
1: -0.0408165, 0.0770043, -0.0301042, 0.0472260, -0.0880425, 0.1071085
2: -0.0973802, 0.1353522, -0.0630851, 0.0951283, -0.1925085, 0.1984372
3: -0.0471661, 0.0854527, -0.0331776, 0.0482606, -0.0954267, 0.1186303
4: -0.0990589, 0.1538413, -0.0595089, 0.1043710, -0.2034298, 0.2133503

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

## BFS NS instance: NS_A1_B1_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0335580, 0.0372201, -0.0238656, 0.0210918, -0.0546498, 0.0610857
1: -0.0408165, 0.0770043, -0.0303834, 0.0492062, -0.0900227, 0.1073877
2: -0.0973802, 0.1353522, -0.0626324, 0.0968737, -0.1942539, 0.1979846
3: -0.0471661, 0.0854527, -0.0338869, 0.0508352, -0.0980013, 0.1193396
4: -0.0990589, 0.1538413, -0.0603797, 0.1074024, -0.2064612, 0.2142210

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

## BFS NS instance: NS_A1_B1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0335580, 0.0372201, -0.0303041, 0.0332229, -0.0667809, 0.0675242
1: -0.0408165, 0.0770043, -0.0381444, 0.0714089, -0.1122254, 0.1151486
2: -0.0973802, 0.1353522, -0.0873048, 0.1254651, -0.2228454, 0.2226568
3: -0.0471661, 0.0854527, -0.0446786, 0.0786664, -0.1258325, 0.1301313
4: -0.0990589, 0.1538413, -0.0888710, 0.1432975, -0.2423564, 0.2427123

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

## BFS NS instance: NS_A1_B1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0335580, 0.0372201, -0.0335580, 0.0372201, -0.0707781, 0.0707781
1: -0.0408165, 0.0770043, -0.0408165, 0.0770043, -0.1178208, 0.1178208
2: -0.0973802, 0.1353522, -0.0973802, 0.1353522, -0.2327324, 0.2327324
3: -0.0471661, 0.0854527, -0.0471661, 0.0854527, -0.1326189, 0.1326188
4: -0.0990589, 0.1538413, -0.0990589, 0.1538413, -0.2529002, 0.2529002

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

## BFS NS instance: NS_A1_B2_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0271122, 0.0295338, -0.0245046, 0.0304916, -0.0576038, 0.0540384
1: -0.0338762, 0.0683478, -0.0370066, 0.0861242, -0.1200004, 0.1053544
2: -0.0775269, 0.1190995, -0.0676304, 0.1160857, -0.1936127, 0.1867298
3: -0.0413828, 0.0783211, -0.0456015, 0.1057874, -0.1471702, 0.1239226
4: -0.0817343, 0.1358244, -0.0840454, 0.1383576, -0.2200919, 0.2198697

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886952, upper bound: 0.0891720
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_A2

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887542, upper bound: 0.0889950
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0280932, 0.0294513, -0.0245046, 0.0304916, -0.0585848, 0.0539559
1: -0.0348121, 0.0680492, -0.0370066, 0.0861242, -0.1209363, 0.1050558
2: -0.0808951, 0.1180537, -0.0676304, 0.1160857, -0.1969808, 0.1856840
3: -0.0416559, 0.0777593, -0.0456015, 0.1057874, -0.1474433, 0.1233608
4: -0.0845517, 0.1341558, -0.0840454, 0.1383576, -0.2229093, 0.2182012

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_A1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886952, upper bound: 0.0891720
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_A2

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887542, upper bound: 0.0889950
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0189038, 0.0158268, -0.0293784, 0.0348040, -0.0537078, 0.0452051
1: -0.0248756, 0.0358514, -0.0411437, 0.0922270, -0.1171026, 0.0769951
2: -0.0468024, 0.0796161, -0.0824305, 0.1296569, -0.1764594, 0.1620466
3: -0.0280160, 0.0339445, -0.0493116, 0.1114420, -0.1394580, 0.0832561
4: -0.0429375, 0.0872777, -0.0963648, 0.1533624, -0.1962999, 0.1836426

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886940, upper bound: 0.0891682
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886940, upper bound: 0.0891682
time: 0.23 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0274795, 0.0253483, -0.0293784, 0.0348040, -0.0622835, 0.0547266
1: -0.0339634, 0.0554254, -0.0411437, 0.0922270, -0.1261904, 0.0965691
2: -0.0789161, 0.1109374, -0.0824305, 0.1296569, -0.2085730, 0.1933679
3: -0.0402230, 0.0574996, -0.0493116, 0.1114420, -0.1516650, 0.1068112
4: -0.0770320, 0.1258029, -0.0963648, 0.1533624, -0.2303945, 0.2221678

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887542, upper bound: 0.0889951
time: 0.25 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887542, upper bound: 0.0889950
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0189038, 0.0158268, -0.0298293, 0.0411537, -0.0600575, 0.0456561
1: -0.0248756, 0.0358514, -0.0437109, 0.1075221, -0.1323977, 0.0795622
2: -0.0468024, 0.0796161, -0.0845616, 0.1400728, -0.1868752, 0.1641776
3: -0.0280160, 0.0339445, -0.0523840, 0.1329535, -0.1609695, 0.0863285
4: -0.0429375, 0.0872777, -0.1043052, 0.1662240, -0.2091615, 0.1915830

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886121, upper bound: 0.0891797
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886121, upper bound: 0.0891797
time: 0.40 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0274795, 0.0253483, -0.0298293, 0.0411537, -0.0686332, 0.0551776
1: -0.0339634, 0.0554254, -0.0437109, 0.1075221, -0.1414855, 0.0991362
2: -0.0789161, 0.1109374, -0.0845616, 0.1400728, -0.2189889, 0.1954990
3: -0.0402230, 0.0574996, -0.0523840, 0.1329535, -0.1731765, 0.1098836
4: -0.0770320, 0.1258029, -0.1043052, 0.1662240, -0.2432560, 0.2301082

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887520, upper bound: 0.0890131
time: 0.25 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887520, upper bound: 0.0890131
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0189038, 0.0158268, -0.0339535, 0.0482027, -0.0671065, 0.0497803
1: -0.0248756, 0.0358514, -0.0470253, 0.1167137, -0.1415893, 0.0828767
2: -0.0468024, 0.0796161, -0.0978653, 0.1553957, -0.2021981, 0.1774814
3: -0.0280160, 0.0339445, -0.0554273, 0.1441097, -0.1721257, 0.0893717
4: -0.0429375, 0.0872777, -0.1172390, 0.1831006, -0.2260380, 0.2045167

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883057, upper bound: 0.0888244
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883057, upper bound: 0.0888558
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0274795, 0.0253483, -0.0339535, 0.0482027, -0.0756822, 0.0593018
1: -0.0339634, 0.0554254, -0.0470253, 0.1167137, -0.1506772, 0.1024507
2: -0.0789161, 0.1109374, -0.0978653, 0.1553957, -0.2343118, 0.2088027
3: -0.0402230, 0.0574996, -0.0554273, 0.1441097, -0.1843327, 0.1129269
4: -0.0770320, 0.1258029, -0.1172390, 0.1831006, -0.2601326, 0.2430419

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887073, upper bound: 0.0890131
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887073, upper bound: 0.0890131
time: 0.23 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0237667, 0.0206726, -0.0221238, 0.0218550, -0.0456217, 0.0427965
1: -0.0301767, 0.0481623, -0.0302679, 0.0559016, -0.0860783, 0.0784302
2: -0.0622640, 0.0955307, -0.0555221, 0.0900413, -0.1523053, 0.1510528
3: -0.0332325, 0.0495364, -0.0350749, 0.0613522, -0.0945847, 0.0846113
4: -0.0597730, 0.1052151, -0.0581665, 0.1034390, -0.1632120, 0.1633816

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0237667, 0.0206726, -0.0268914, 0.0293539, -0.0531206, 0.0475640
1: -0.0301767, 0.0481623, -0.0379840, 0.0740987, -0.1042754, 0.0861463
2: -0.0622640, 0.0955307, -0.0746911, 0.1158494, -0.1781134, 0.1702218
3: -0.0332325, 0.0495364, -0.0449665, 0.0842463, -0.1174788, 0.0945029
4: -0.0597730, 0.1052151, -0.0825456, 0.1361845, -0.1959575, 0.1877607

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0322836, 0.0334207, -0.0221238, 0.0218550, -0.0541385, 0.0555445
1: -0.0396060, 0.0710703, -0.0302679, 0.0559016, -0.0955077, 0.1013381
2: -0.0933639, 0.1287424, -0.0555221, 0.0900413, -0.1834052, 0.1842645
3: -0.0451630, 0.0779622, -0.0350749, 0.0613522, -0.1065152, 0.1130371
4: -0.0942796, 0.1455511, -0.0581665, 0.1034390, -0.1977186, 0.2037176

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0322836, 0.0334207, -0.0268914, 0.0293539, -0.0616375, 0.0603120
1: -0.0396060, 0.0710703, -0.0379840, 0.0740987, -0.1137047, 0.1090542
2: -0.0933639, 0.1287424, -0.0746911, 0.1158494, -0.2092133, 0.2034335
3: -0.0451630, 0.0779622, -0.0449665, 0.0842463, -0.1294093, 0.1229287
4: -0.0942796, 0.1455511, -0.0825456, 0.1361845, -0.2304641, 0.2280967

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0237667, 0.0206726, -0.0267372, 0.0278463, -0.0516130, 0.0474098
1: -0.0301767, 0.0481623, -0.0364584, 0.0733053, -0.1034820, 0.0846208
2: -0.0622640, 0.0955307, -0.0708168, 0.1098818, -0.1721458, 0.1663475
3: -0.0332325, 0.0495364, -0.0413998, 0.0853549, -0.1185874, 0.0909362
4: -0.0597730, 0.1052151, -0.0783475, 0.1270356, -0.1868086, 0.1835625

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0237667, 0.0206726, -0.0329713, 0.0419844, -0.0657511, 0.0536439
1: -0.0301767, 0.0481623, -0.0460162, 0.0978074, -0.1279841, 0.0941785
2: -0.0622640, 0.0955307, -0.0935178, 0.1428516, -0.2051156, 0.1890485
3: -0.0332325, 0.0495364, -0.0521133, 0.1155885, -0.1488211, 0.1016497
4: -0.0597730, 0.1052151, -0.1060875, 0.1669992, -0.2267722, 0.2113025

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0322836, 0.0334207, -0.0267372, 0.0278463, -0.0601299, 0.0601578
1: -0.0396060, 0.0710703, -0.0364584, 0.0733053, -0.1129114, 0.1075287
2: -0.0933639, 0.1287424, -0.0708168, 0.1098818, -0.2032457, 0.1995592
3: -0.0451630, 0.0779622, -0.0413998, 0.0853549, -0.1305179, 0.1193620
4: -0.0942796, 0.1455511, -0.0783475, 0.1270356, -0.2213152, 0.2238985

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0322836, 0.0334207, -0.0329713, 0.0419844, -0.0742679, 0.0663919
1: -0.0396060, 0.0710703, -0.0460162, 0.0978074, -0.1374134, 0.1170865
2: -0.0933639, 0.1287424, -0.0935178, 0.1428516, -0.2362154, 0.2222602
3: -0.0451630, 0.0779622, -0.0521133, 0.1155885, -0.1607515, 0.1300755
4: -0.0942796, 0.1455511, -0.1060875, 0.1669992, -0.2612788, 0.2516386

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A2_A1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0215165, 0.0264666, -0.0288182, 0.0320959, -0.0536124, 0.0552847
1: -0.0332614, 0.0721512, -0.0419773, 0.0852667, -0.1185281, 0.1141285
2: -0.0573547, 0.1032774, -0.0796866, 0.1238025, -0.1811572, 0.1829639
3: -0.0413794, 0.0860633, -0.0488805, 0.1012414, -0.1426207, 0.1349438
4: -0.0701147, 0.1229001, -0.0916779, 0.1468575, -0.2169722, 0.2145780

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A1_B1_A1_B1_B1

### Relational analysis result of NS_A2_A1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898256, upper bound: 0.0898481
time: 0.24 seconds

## Relational analysis of NS_A2_A1_A1_B1_A1_B1_B2

### Relational analysis result of NS_A2_A1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898256, upper bound: 0.0898560
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0215165, 0.0264666, -0.0335036, 0.0434306, -0.0649472, 0.0599702
1: -0.0332614, 0.0721512, -0.0478994, 0.1090685, -0.1423299, 0.1200505
2: -0.0573547, 0.1032774, -0.0958872, 0.1491258, -0.2064804, 0.1991646
3: -0.0413794, 0.0860633, -0.0553259, 0.1333559, -0.1747352, 0.1413892
4: -0.0701147, 0.1229001, -0.1132077, 0.1763901, -0.2465048, 0.2361078

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A1_B1_A1_B2_B1

### Relational analysis result of NS_A2_A1_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898256, upper bound: 0.0898724
time: 0.25 seconds

## Relational analysis of NS_A2_A1_A1_B1_A1_B2_B2

### Relational analysis result of NS_A2_A1_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898256, upper bound: 0.0898724
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0271535, 0.0296738, -0.0288182, 0.0320959, -0.0592494, 0.0584920
1: -0.0381406, 0.0776904, -0.0419773, 0.0852667, -0.1234073, 0.1196677
2: -0.0742795, 0.1164777, -0.0796866, 0.1238025, -0.1980821, 0.1961643
3: -0.0456244, 0.0911491, -0.0488805, 0.1012414, -0.1468657, 0.1400296
4: -0.0842188, 0.1373387, -0.0916779, 0.1468575, -0.2310763, 0.2290166

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A1_B1_A2_B1_B1

### Relational analysis result of NS_A2_A1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893923, upper bound: 0.0898440
time: 0.26 seconds

## Relational analysis of NS_A2_A1_A1_B1_A2_B1_B2

### Relational analysis result of NS_A2_A1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898335, upper bound: 0.0898440
time: 0.24 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0271535, 0.0296738, -0.0335036, 0.0434306, -0.0705841, 0.0631775
1: -0.0381406, 0.0776904, -0.0478994, 0.1090685, -0.1472092, 0.1255898
2: -0.0742795, 0.1164777, -0.0958872, 0.1491258, -0.2234053, 0.2123649
3: -0.0456244, 0.0911491, -0.0553259, 0.1333559, -0.1789802, 0.1464750
4: -0.0842188, 0.1373387, -0.1132077, 0.1763901, -0.2606089, 0.2505464

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A1_B1_A2_B2_B1

### Relational analysis result of NS_A2_A1_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898335, upper bound: 0.0898683
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A1_B1_A2_B2_B2

### Relational analysis result of NS_A2_A1_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898335, upper bound: 0.0898683
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0204322, 0.0194142, -0.0264032, 0.0510042, -0.0714364, 0.0458174
1: -0.0283589, 0.0475317, -0.0573585, 0.1537825, -0.1821414, 0.1048902
2: -0.0497279, 0.0810393, -0.0757487, 0.1525275, -0.2022554, 0.1567880
3: -0.0324287, 0.0497364, -0.0655027, 0.2003427, -0.2327714, 0.1152390
4: -0.0500239, 0.0922334, -0.1182454, 0.1901293, -0.2401533, 0.2104788

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A1

### Relational analysis result of NS_A2_A1_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881385, upper bound: 0.0889542
time: 0.25 seconds

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A2

### Relational analysis result of NS_A2_A1_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881385, upper bound: 0.0889542
time: 0.25 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0232749, 0.0259113, -0.0264032, 0.0510042, -0.0742791, 0.0523145
1: -0.0340355, 0.0645256, -0.0573585, 0.1537825, -0.1878180, 0.1218841
2: -0.0633950, 0.1038888, -0.0757487, 0.1525275, -0.2159224, 0.1796376
3: -0.0410979, 0.0714647, -0.0655027, 0.2003427, -0.2414406, 0.1369673
4: -0.0702281, 0.1223172, -0.1182454, 0.1901293, -0.2603574, 0.2405626

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A1

### Relational analysis result of NS_A2_A1_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881298, upper bound: 0.0891578
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A2

### Relational analysis result of NS_A2_A1_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881298, upper bound: 0.0892236
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0204322, 0.0194142, -0.0275337, 0.0522360, -0.0726682, 0.0469479
1: -0.0283589, 0.0475317, -0.0577662, 0.1554659, -0.1838248, 0.1052979
2: -0.0497279, 0.0810393, -0.0781244, 0.1540693, -0.2037973, 0.1591637
3: -0.0324287, 0.0497364, -0.0657998, 0.2024251, -0.2348538, 0.1155362
4: -0.0500239, 0.0922334, -0.1204030, 0.1915261, -0.2415500, 0.2126364

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_A1_A1_B2_B2_A1_B1

### Relational analysis result of NS_A2_A1_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877640, upper bound: 0.0887309
time: 0.25 seconds

## Relational analysis of NS_A2_A1_A1_B2_B2_A1_B2

### Relational analysis result of NS_A2_A1_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877640, upper bound: 0.0889656
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0232749, 0.0259113, -0.0275337, 0.0522360, -0.0755109, 0.0534450
1: -0.0340355, 0.0645256, -0.0577662, 0.1554659, -0.1895014, 0.1222918
2: -0.0633950, 0.1038888, -0.0781244, 0.1540693, -0.2174643, 0.1820132
3: -0.0410979, 0.0714647, -0.0657998, 0.2024251, -0.2435230, 0.1372645
4: -0.0702281, 0.1223172, -0.1204030, 0.1915261, -0.2617542, 0.2427202

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_A1_A1_B2_B2_A2_B1

### Relational analysis result of NS_A2_A1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877674, upper bound: 0.0888610
time: 0.25 seconds

## Relational analysis of NS_A2_A1_A1_B2_B2_A2_B2

### Relational analysis result of NS_A2_A1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877674, upper bound: 0.0888610
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0326413, 0.0406024, -0.0221645, 0.0274029, -0.0600441, 0.0627669
1: -0.0464028, 0.1029233, -0.0345445, 0.0750418, -0.1214446, 0.1374678
2: -0.0931736, 0.1439410, -0.0594218, 0.1060993, -0.1992729, 0.2033627
3: -0.0537068, 0.1251042, -0.0427037, 0.0898498, -0.1435566, 0.1678079
4: -0.1092366, 0.1700594, -0.0729370, 0.1265663, -0.2358029, 0.2429964

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A2_B1_B1_B1_A1

### Relational analysis result of NS_A2_A1_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898683, upper bound: 0.0898256
time: 0.25 seconds

## Relational analysis of NS_A2_A1_A2_B1_B1_B1_A2

### Relational analysis result of NS_A2_A1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898683, upper bound: 0.0898256
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0326413, 0.0406024, -0.0278082, 0.0306353, -0.0632766, 0.0684105
1: -0.0464028, 0.1029233, -0.0394648, 0.0807291, -0.1271319, 0.1423880
2: -0.0931736, 0.1439410, -0.0763568, 0.1193576, -0.2125312, 0.2202978
3: -0.0537068, 0.1251042, -0.0468939, 0.0951927, -0.1488995, 0.1719981
4: -0.1092366, 0.1700594, -0.0870710, 0.1410980, -0.2503346, 0.2571304

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A2_B1_B1_B2_A1

### Relational analysis result of NS_A2_A1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898683, upper bound: 0.0898335
time: 0.25 seconds

## Relational analysis of NS_A2_A1_A2_B1_B1_B2_A2

### Relational analysis result of NS_A2_A1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898683, upper bound: 0.0898335
time: 0.25 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0261405, 0.0324750, -0.0335036, 0.0434306, -0.0695712, 0.0659786
1: -0.0392807, 0.0896990, -0.0478994, 0.1090685, -0.1483492, 0.1375984
2: -0.0727650, 0.1226256, -0.0958872, 0.1491258, -0.2218907, 0.2185128
3: -0.0478143, 0.1099838, -0.0553259, 0.1333559, -0.1811702, 0.1653097
4: -0.0895612, 0.1459680, -0.1132077, 0.1763901, -0.2659513, 0.2591757

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_A1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898499, upper bound: 0.0898279
time: 0.28 seconds

## Relational analysis of NS_A2_A1_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_A1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898499, upper bound: 0.0898335
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0312804, 0.0368311, -0.0335036, 0.0434306, -0.0747110, 0.0703348
1: -0.0436182, 0.0962436, -0.0478994, 0.1090685, -0.1526867, 0.1441430
2: -0.0892005, 0.1371617, -0.0958872, 0.1491258, -0.2383263, 0.2330489
3: -0.0514396, 0.1168116, -0.0553259, 0.1333559, -0.1847954, 0.1721376
4: -0.1041426, 0.1617303, -0.1132077, 0.1763901, -0.2805327, 0.2749380

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A2_B1_B2_A2_B1

### Relational analysis result of NS_A2_A1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898499, upper bound: 0.0898279
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A2_B1_B2_A2_B2

### Relational analysis result of NS_A2_A1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898499, upper bound: 0.0898335
time: 0.29 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0248626, 0.0250574, -0.0264032, 0.0510042, -0.0758668, 0.0514606
1: -0.0340729, 0.0640337, -0.0573585, 0.1537825, -0.1878553, 0.1213921
2: -0.0645417, 0.1001453, -0.0757487, 0.1525275, -0.2170691, 0.1758940
3: -0.0381859, 0.0723889, -0.0655027, 0.2003427, -0.2385286, 0.1378915
4: -0.0690484, 0.1149325, -0.1182454, 0.1901293, -0.2591777, 0.2331779

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_A1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A2_B2_B1_A1_A1

### Relational analysis result of NS_A2_A1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881544, upper bound: 0.0889778
time: 0.24 seconds

## Relational analysis of NS_A2_A1_A2_B2_B1_A1_A2

### Relational analysis result of NS_A2_A1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881544, upper bound: 0.0889778
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0287746, 0.0323239, -0.0264032, 0.0510042, -0.0797788, 0.0587272
1: -0.0413172, 0.0814996, -0.0573585, 0.1537825, -0.1950997, 0.1388581
2: -0.0808285, 0.1242037, -0.0757487, 0.1525275, -0.2333559, 0.1999524
3: -0.0476249, 0.0947150, -0.0655027, 0.2003427, -0.2479676, 0.1602177
4: -0.0914771, 0.1458504, -0.1182454, 0.1901293, -0.2816064, 0.2640958

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_A1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A2_B2_B1_A2_A1

### Relational analysis result of NS_A2_A1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881638, upper bound: 0.0892348
time: 0.26 seconds

## Relational analysis of NS_A2_A1_A2_B2_B1_A2_A2

### Relational analysis result of NS_A2_A1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881638, upper bound: 0.0892348
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0248626, 0.0250574, -0.0275337, 0.0522360, -0.0770986, 0.0525911
1: -0.0340729, 0.0640337, -0.0577662, 0.1554659, -0.1895388, 0.1217998
2: -0.0645417, 0.1001453, -0.0781244, 0.1540693, -0.2186110, 0.1782697
3: -0.0381859, 0.0723889, -0.0657998, 0.2024251, -0.2406110, 0.1381887
4: -0.0690484, 0.1149325, -0.1204030, 0.1915261, -0.2605745, 0.2353355

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_A1_A2_B2_B2_A1_B1

### Relational analysis result of NS_A2_A1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877759, upper bound: 0.0887592
time: 0.28 seconds

## Relational analysis of NS_A2_A1_A2_B2_B2_A1_B2

### Relational analysis result of NS_A2_A1_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877759, upper bound: 0.0887592
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0287746, 0.0323239, -0.0275337, 0.0522360, -0.0810106, 0.0598577
1: -0.0413172, 0.0814996, -0.0577662, 0.1554659, -0.1967831, 0.1392658
2: -0.0808285, 0.1242037, -0.0781244, 0.1540693, -0.2348978, 0.2023280
3: -0.0476249, 0.0947150, -0.0657998, 0.2024251, -0.2500500, 0.1605148
4: -0.0914771, 0.1458504, -0.1204030, 0.1915261, -0.2830032, 0.2662534

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_A1_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_A1_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877853, upper bound: 0.0889125
time: 0.24 seconds

## Relational analysis of NS_A2_A1_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_A1_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877853, upper bound: 0.0889125
time: 0.26 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 1.30 seconds
NS_A1_B2_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0886952, upper bound: 0.0891720
NS_A1_B2_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0887542, upper bound: 0.0889950
NS_A1_B2_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0886952, upper bound: 0.0891720
NS_A1_B2_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0887542, upper bound: 0.0889950
NS_A1_B2_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0886940, upper bound: 0.0891682
NS_A1_B2_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0886940, upper bound: 0.0891682
NS_A1_B2_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0887542, upper bound: 0.0889951
NS_A1_B2_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0887542, upper bound: 0.0889950
NS_A1_B2_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0886121, upper bound: 0.0891797
NS_A1_B2_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0886121, upper bound: 0.0891797
NS_A1_B2_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0887520, upper bound: 0.0890131
NS_A1_B2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0887520, upper bound: 0.0890131
NS_A1_B2_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0883057, upper bound: 0.0888244
NS_A1_B2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0883057, upper bound: 0.0888558
NS_A1_B2_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0887073, upper bound: 0.0890131
NS_A1_B2_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0887073, upper bound: 0.0890131
NS_A2_A1_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0898256, upper bound: 0.0898481
NS_A2_A1_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0898256, upper bound: 0.0898560
NS_A2_A1_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0898256, upper bound: 0.0898724
NS_A2_A1_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0898256, upper bound: 0.0898724
NS_A2_A1_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0893923, upper bound: 0.0898440
NS_A2_A1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0898335, upper bound: 0.0898440
NS_A2_A1_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0898335, upper bound: 0.0898683
NS_A2_A1_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0898335, upper bound: 0.0898683
NS_A2_A1_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0881385, upper bound: 0.0889542
NS_A2_A1_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0881385, upper bound: 0.0889542
NS_A2_A1_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0881298, upper bound: 0.0891578
NS_A2_A1_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0881298, upper bound: 0.0892236
NS_A2_A1_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0877640, upper bound: 0.0887309
NS_A2_A1_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0877640, upper bound: 0.0889656
NS_A2_A1_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0877674, upper bound: 0.0888610
NS_A2_A1_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0877674, upper bound: 0.0888610
NS_A2_A1_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0898683, upper bound: 0.0898256
NS_A2_A1_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0898683, upper bound: 0.0898256
NS_A2_A1_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0898683, upper bound: 0.0898335
NS_A2_A1_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0898683, upper bound: 0.0898335
NS_A2_A1_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0898499, upper bound: 0.0898279
NS_A2_A1_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0898499, upper bound: 0.0898335
NS_A2_A1_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0898499, upper bound: 0.0898279
NS_A2_A1_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0898499, upper bound: 0.0898335
NS_A2_A1_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0881544, upper bound: 0.0889778
NS_A2_A1_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0881544, upper bound: 0.0889778
NS_A2_A1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0881638, upper bound: 0.0892348
NS_A2_A1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0881638, upper bound: 0.0892348
NS_A2_A1_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0877759, upper bound: 0.0887592
NS_A2_A1_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0877759, upper bound: 0.0887592
NS_A2_A1_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0877853, upper bound: 0.0889125
NS_A2_A1_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 1.30
Output dim: 0, lower bound: -0.0877853, upper bound: 0.0889125

## BFS NS instance: NS_A1_B2_A1_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0183061, 0.0140802, -0.0245046, 0.0304916, -0.0487977, 0.0385848
1: -0.0240727, 0.0309017, -0.0370066, 0.0861242, -0.1101969, 0.0679083
2: -0.0450367, 0.0758257, -0.0676304, 0.1160857, -0.1611224, 0.1434560
3: -0.0263939, 0.0274213, -0.0456015, 0.1057874, -0.1321813, 0.0730227
4: -0.0390864, 0.0815583, -0.0840454, 0.1383576, -0.1774440, 0.1656037

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885484, upper bound: 0.0888717
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884127, upper bound: 0.0884127
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0235625, 0.0223077, -0.0245046, 0.0304916, -0.0540542, 0.0468123
1: -0.0298257, 0.0489782, -0.0370066, 0.0861242, -0.1159499, 0.0859848
2: -0.0665481, 0.0995724, -0.0676304, 0.1160857, -0.1826338, 0.1672027
3: -0.0366166, 0.0498813, -0.0456015, 0.1057874, -0.1424040, 0.0954828
4: -0.0645988, 0.1130393, -0.0840454, 0.1383576, -0.2029564, 0.1970846

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885484, upper bound: 0.0890128
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885484, upper bound: 0.0890429
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0182536, 0.0148930, -0.0245046, 0.0304916, -0.0487452, 0.0393976
1: -0.0242180, 0.0328669, -0.0370066, 0.0861242, -0.1103422, 0.0698735
2: -0.0444218, 0.0763963, -0.0676304, 0.1160857, -0.1605076, 0.1440267
3: -0.0269134, 0.0300028, -0.0456015, 0.1057874, -0.1327008, 0.0756043
4: -0.0399020, 0.0830934, -0.0840454, 0.1383576, -0.1782596, 0.1671388

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883016, upper bound: 0.0888076
time: 0.25 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883016, upper bound: 0.0888377
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0268614, 0.0244703, -0.0245046, 0.0304916, -0.0573530, 0.0489749
1: -0.0328907, 0.0524811, -0.0370066, 0.0861242, -0.1190149, 0.0894877
2: -0.0769134, 0.1082844, -0.0676304, 0.1160857, -0.1929991, 0.1759147
3: -0.0390857, 0.0535550, -0.0456015, 0.1057874, -0.1448731, 0.0991565
4: -0.0741939, 0.1224238, -0.0840454, 0.1383576, -0.2125515, 0.2064691

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883016, upper bound: 0.0889649
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A1_B1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883016, upper bound: 0.0889950
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0183061, 0.0140802, -0.0293784, 0.0348040, -0.0531100, 0.0434586
1: -0.0240727, 0.0309017, -0.0411437, 0.0922270, -0.1162997, 0.0720454
2: -0.0450367, 0.0758257, -0.0824305, 0.1296569, -0.1746936, 0.1582562
3: -0.0263939, 0.0274213, -0.0493116, 0.1114420, -0.1378358, 0.0767328
4: -0.0390864, 0.0815583, -0.0963648, 0.1533624, -0.1924489, 0.1779232

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884123, upper bound: 0.0885925
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885889, upper bound: 0.0891682
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0182536, 0.0148930, -0.0293784, 0.0348040, -0.0530576, 0.0442713
1: -0.0242180, 0.0328669, -0.0411437, 0.0922270, -0.1164450, 0.0740106
2: -0.0444218, 0.0763963, -0.0824305, 0.1296569, -0.1740788, 0.1588268
3: -0.0269134, 0.0300028, -0.0493116, 0.1114420, -0.1383554, 0.0793144
4: -0.0399020, 0.0830934, -0.0963648, 0.1533624, -0.1932644, 0.1794583

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885889, upper bound: 0.0891682
time: 0.24 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885889, upper bound: 0.0891682
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0235625, 0.0223077, -0.0293784, 0.0348040, -0.0583665, 0.0516860
1: -0.0298257, 0.0489782, -0.0411437, 0.0922270, -0.1220528, 0.0901219
2: -0.0665481, 0.0995724, -0.0824305, 0.1296569, -0.1962050, 0.1820029
3: -0.0366166, 0.0498813, -0.0493116, 0.1114420, -0.1480585, 0.0991929
4: -0.0645988, 0.1130393, -0.0963648, 0.1533624, -0.2179612, 0.2094041

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883016, upper bound: 0.0889706
time: 0.25 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883016, upper bound: 0.0889951
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0268614, 0.0244703, -0.0293784, 0.0348040, -0.0616654, 0.0538487
1: -0.0328907, 0.0524811, -0.0411437, 0.0922270, -0.1251177, 0.0936248
2: -0.0769134, 0.1082844, -0.0824305, 0.1296569, -0.2065703, 0.1907149
3: -0.0390857, 0.0535550, -0.0493116, 0.1114420, -0.1505277, 0.1028665
4: -0.0741939, 0.1224238, -0.0963648, 0.1533624, -0.2275563, 0.2187886

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883016, upper bound: 0.0889649
time: 0.25 seconds

## Relational analysis of NS_A1_B2_A1_B1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883016, upper bound: 0.0889950
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0183061, 0.0140802, -0.0298293, 0.0411537, -0.0594598, 0.0439095
1: -0.0240727, 0.0309017, -0.0437109, 0.1075221, -0.1315947, 0.0746125
2: -0.0450367, 0.0758257, -0.0845616, 0.1400728, -0.1851095, 0.1603873
3: -0.0263939, 0.0274213, -0.0523840, 0.1329535, -0.1593473, 0.0798052
4: -0.0390864, 0.0815583, -0.1043052, 0.1662240, -0.2053104, 0.1858636

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884765, upper bound: 0.0891797
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884765, upper bound: 0.0891797
time: 0.24 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0182536, 0.0148930, -0.0298293, 0.0411537, -0.0594073, 0.0447223
1: -0.0242180, 0.0328669, -0.0437109, 0.1075221, -0.1317401, 0.0765778
2: -0.0444218, 0.0763963, -0.0845616, 0.1400728, -0.1844946, 0.1609579
3: -0.0269134, 0.0300028, -0.0523840, 0.1329535, -0.1598669, 0.0823868
4: -0.0399020, 0.0830934, -0.1043052, 0.1662240, -0.2061260, 0.1873987

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884765, upper bound: 0.0891797
time: 0.25 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884765, upper bound: 0.0891797
time: 0.28 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0235625, 0.0223077, -0.0298293, 0.0411537, -0.0647163, 0.0521369
1: -0.0298257, 0.0489782, -0.0437109, 0.1075221, -0.1373478, 0.0926891
2: -0.0665481, 0.0995724, -0.0845616, 0.1400728, -0.2066209, 0.1841339
3: -0.0366166, 0.0498813, -0.0523840, 0.1329535, -0.1695700, 0.1022653
4: -0.0645988, 0.1130393, -0.1043052, 0.1662240, -0.2308228, 0.2173445

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883148, upper bound: 0.0889823
time: 0.25 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881659, upper bound: 0.0890131
time: 0.25 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0268614, 0.0244703, -0.0298293, 0.0411537, -0.0680152, 0.0542996
1: -0.0328907, 0.0524811, -0.0437109, 0.1075221, -0.1404127, 0.0961920
2: -0.0769134, 0.1082844, -0.0845616, 0.1400728, -0.2169862, 0.1928460
3: -0.0390857, 0.0535550, -0.0523840, 0.1329535, -0.1720392, 0.1059390
4: -0.0741939, 0.1224238, -0.1043052, 0.1662240, -0.2404179, 0.2267290

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881659, upper bound: 0.0885881
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883148, upper bound: 0.0890131
time: 0.26 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0189038, 0.0158268, -0.0254340, 0.0263058, -0.0452096, 0.0412608
1: -0.0248756, 0.0358514, -0.0336134, 0.0685003, -0.0933759, 0.0694647
2: -0.0468024, 0.0796161, -0.0668550, 0.1052071, -0.1520095, 0.1464711
3: -0.0280160, 0.0339445, -0.0392151, 0.0791419, -0.1071579, 0.0731595
4: -0.0429375, 0.0872777, -0.0734634, 0.1209796, -0.1639170, 0.1607412

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A1_B2_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0189038, 0.0158268, -0.0319843, 0.0386605, -0.0575643, 0.0478111
1: -0.0248756, 0.0358514, -0.0438460, 0.0911426, -0.1160182, 0.0796974
2: -0.0468024, 0.0796161, -0.0905300, 0.1369953, -0.1837978, 0.1701460
3: -0.0280160, 0.0339445, -0.0502524, 0.1067187, -0.1347347, 0.0841969
4: -0.0429375, 0.0872777, -0.1015165, 0.1598046, -0.2027421, 0.1887942

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0235625, 0.0223077, -0.0339535, 0.0482027, -0.0717652, 0.0562612
1: -0.0298257, 0.0489782, -0.0470253, 0.1167137, -0.1465395, 0.0960035
2: -0.0665481, 0.0995724, -0.0978653, 0.1553957, -0.2219437, 0.1974377
3: -0.0366166, 0.0498813, -0.0554273, 0.1441097, -0.1807263, 0.1053086
4: -0.0645988, 0.1130393, -0.1172390, 0.1831006, -0.2476994, 0.2302783

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883057, upper bound: 0.0889817
time: 0.26 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883057, upper bound: 0.0890131
time: 0.26 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0268614, 0.0244703, -0.0339535, 0.0482027, -0.0750641, 0.0584239
1: -0.0328907, 0.0524811, -0.0470253, 0.1167137, -0.1496044, 0.0995064
2: -0.0769134, 0.1082844, -0.0978653, 0.1553957, -0.2323091, 0.2061497
3: -0.0390857, 0.0535550, -0.0554273, 0.1441097, -0.1831955, 0.1089822
4: -0.0741939, 0.1224238, -0.1172390, 0.1831006, -0.2572944, 0.2396628

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883057, upper bound: 0.0889817
time: 0.25 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883057, upper bound: 0.0890131
time: 0.24 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0215165, 0.0264666, -0.0221645, 0.0274029, -0.0489194, 0.0486311
1: -0.0332614, 0.0721512, -0.0345445, 0.0750418, -0.1083031, 0.1066957
2: -0.0573547, 0.1032774, -0.0594218, 0.1060993, -0.1634540, 0.1626991
3: -0.0413794, 0.0860633, -0.0427037, 0.0898498, -0.1312291, 0.1287670
4: -0.0701147, 0.1229001, -0.0729370, 0.1265663, -0.1966810, 0.1958371

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A1_B1_A1_B1_B1_A1

### Relational analysis result of NS_A2_A1_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894541, upper bound: 0.0894492
time: 0.24 seconds

## Relational analysis of NS_A2_A1_A1_B1_A1_B1_B1_A2

### Relational analysis result of NS_A2_A1_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894454, upper bound: 0.0894453
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0215165, 0.0264666, -0.0278082, 0.0306353, -0.0521519, 0.0542747
1: -0.0332614, 0.0721512, -0.0394648, 0.0807291, -0.1139905, 0.1116159
2: -0.0573547, 0.1032774, -0.0763568, 0.1193576, -0.1767123, 0.1796342
3: -0.0413794, 0.0860633, -0.0468939, 0.0951927, -0.1365720, 0.1329572
4: -0.0701147, 0.1229001, -0.0870710, 0.1410980, -0.2112127, 0.2099711

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A1_B1_A1_B1_B2_B1

### Relational analysis result of NS_A2_A1_A1_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894453, upper bound: 0.0894573
time: 0.23 seconds

## Relational analysis of NS_A2_A1_A1_B1_A1_B1_B2_B2

### Relational analysis result of NS_A2_A1_A1_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894453, upper bound: 0.0894615
time: 0.23 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0215165, 0.0264666, -0.0268901, 0.0335713, -0.0550878, 0.0533566
1: -0.0332614, 0.0721512, -0.0407071, 0.0941096, -0.1273710, 0.1128582
2: -0.0573547, 0.1032774, -0.0752188, 0.1259669, -0.1833216, 0.1784962
3: -0.0413794, 0.0860633, -0.0493354, 0.1159112, -0.1572906, 0.1353986
4: -0.0701147, 0.1229001, -0.0929475, 0.1502719, -0.2203866, 0.2158477

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A1_B1_A1_B2_B1_B1

### Relational analysis result of NS_A2_A1_A1_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894345, upper bound: 0.0894647
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A1_B1_A1_B2_B1_B2

### Relational analysis result of NS_A2_A1_A1_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893945, upper bound: 0.0894794
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0215165, 0.0264666, -0.0321185, 0.0395926, -0.0611091, 0.0585850
1: -0.0332614, 0.0721512, -0.0450869, 0.1022720, -0.1355334, 0.1172381
2: -0.0573547, 0.1032774, -0.0918327, 0.1422157, -0.1995704, 0.1951101
3: -0.0413794, 0.0860633, -0.0529834, 0.1246344, -0.1660138, 0.1390467
4: -0.0701147, 0.1229001, -0.1079243, 0.1679169, -0.2380316, 0.2308244

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A1_B1_A1_B2_B2_B1

### Relational analysis result of NS_A2_A1_A1_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894345, upper bound: 0.0894647
time: 0.26 seconds

## Relational analysis of NS_A2_A1_A1_B1_A1_B2_B2_B2

### Relational analysis result of NS_A2_A1_A1_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893945, upper bound: 0.0894794
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0271535, 0.0296738, -0.0221645, 0.0274029, -0.0545564, 0.0518384
1: -0.0381406, 0.0776904, -0.0345445, 0.0750418, -0.1131824, 0.1122349
2: -0.0742795, 0.1164777, -0.0594218, 0.1060993, -0.1803789, 0.1758994
3: -0.0456244, 0.0911491, -0.0427037, 0.0898498, -0.1354741, 0.1338528
4: -0.0842188, 0.1373387, -0.0729370, 0.1265663, -0.2107851, 0.2102757

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A1_B1_A2_B1_B1_A1

### Relational analysis result of NS_A2_A1_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886072, upper bound: 0.0894442
time: 0.24 seconds

## Relational analysis of NS_A2_A1_A1_B1_A2_B1_B1_A2

### Relational analysis result of NS_A2_A1_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894615, upper bound: 0.0894453
time: 0.25 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0271535, 0.0296738, -0.0278082, 0.0306353, -0.0577888, 0.0574820
1: -0.0381406, 0.0776904, -0.0394648, 0.0807291, -0.1188697, 0.1171552
2: -0.0742795, 0.1164777, -0.0763568, 0.1193576, -0.1936372, 0.1928345
3: -0.0456244, 0.0911491, -0.0468939, 0.0951927, -0.1408170, 0.1380430
4: -0.0842188, 0.1373387, -0.0870710, 0.1410980, -0.2253168, 0.2244097

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A1_B1_A2_B1_B2_A1

### Relational analysis result of NS_A2_A1_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894573, upper bound: 0.0894442
time: 0.26 seconds

## Relational analysis of NS_A2_A1_A1_B1_A2_B1_B2_A2

### Relational analysis result of NS_A2_A1_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894615, upper bound: 0.0894487
time: 0.32 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0271535, 0.0296738, -0.0268901, 0.0335713, -0.0607248, 0.0565639
1: -0.0381406, 0.0776904, -0.0407071, 0.0941096, -0.1322502, 0.1183975
2: -0.0742795, 0.1164777, -0.0752188, 0.1259669, -0.2002465, 0.1916965
3: -0.0456244, 0.0911491, -0.0493354, 0.1159112, -0.1615356, 0.1404845
4: -0.0842188, 0.1373387, -0.0929475, 0.1502719, -0.2344907, 0.2302863

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A1_B1_A2_B2_B1_B1

### Relational analysis result of NS_A2_A1_A1_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894338, upper bound: 0.0894647
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A1_B1_A2_B2_B1_B2

### Relational analysis result of NS_A2_A1_A1_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894107, upper bound: 0.0894794
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0271535, 0.0296738, -0.0321185, 0.0395926, -0.0667461, 0.0617923
1: -0.0381406, 0.0776904, -0.0450869, 0.1022720, -0.1404126, 0.1227773
2: -0.0742795, 0.1164777, -0.0918327, 0.1422157, -0.2164952, 0.2083104
3: -0.0456244, 0.0911491, -0.0529834, 0.1246344, -0.1702588, 0.1441325
4: -0.0842188, 0.1373387, -0.1079243, 0.1679169, -0.2521357, 0.2452630

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A1_B1_A2_B2_B2_B1

### Relational analysis result of NS_A2_A1_A1_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885372, upper bound: 0.0894647
time: 0.25 seconds

## Relational analysis of NS_A2_A1_A1_B1_A2_B2_B2_B2

### Relational analysis result of NS_A2_A1_A1_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894107, upper bound: 0.0894794
time: 0.29 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0139696, 0.0151481, -0.0264032, 0.0510042, -0.0649738, 0.0415513
1: -0.0210482, 0.0384102, -0.0573585, 0.1537825, -0.1748306, 0.0957686
2: -0.0298601, 0.0651156, -0.0757487, 0.1525275, -0.1823876, 0.1408643
3: -0.0267865, 0.0398051, -0.0655027, 0.2003427, -0.2271292, 0.1053077
4: -0.0337018, 0.0741010, -0.1182454, 0.1901293, -0.2238311, 0.1923464

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A1_A1

### Relational analysis result of NS_A2_A1_A1_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880282, upper bound: 0.0885835
time: 0.26 seconds

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A1_A2

### Relational analysis result of NS_A2_A1_A1_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880235, upper bound: 0.0885306
time: 0.24 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0195566, 0.0182853, -0.0264032, 0.0510042, -0.0705608, 0.0446885
1: -0.0267812, 0.0438515, -0.0573585, 0.1537825, -0.1805637, 0.1012100
2: -0.0469788, 0.0774942, -0.0757487, 0.1525275, -0.1995063, 0.1532429
3: -0.0308810, 0.0449339, -0.0655027, 0.2003427, -0.2312238, 0.1104365
4: -0.0465213, 0.0876333, -0.1182454, 0.1901293, -0.2366507, 0.2058786

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A2_A1

### Relational analysis result of NS_A2_A1_A1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880282, upper bound: 0.0885835
time: 0.29 seconds

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A2_A2

### Relational analysis result of NS_A2_A1_A1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880235, upper bound: 0.0885306
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0186973, 0.0218708, -0.0264032, 0.0510042, -0.0697015, 0.0482740
1: -0.0282678, 0.0553422, -0.0573585, 0.1537825, -0.1820503, 0.1127007
2: -0.0487225, 0.0898188, -0.0757487, 0.1525275, -0.2012499, 0.1655676
3: -0.0360911, 0.0603832, -0.0655027, 0.2003427, -0.2364338, 0.1258859
4: -0.0556071, 0.1057873, -0.1182454, 0.1901293, -0.2457364, 0.2240327

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A1_A1

### Relational analysis result of NS_A2_A1_A1_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880030, upper bound: 0.0887923
time: 0.23 seconds

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A1_A2

### Relational analysis result of NS_A2_A1_A1_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880195, upper bound: 0.0889269
time: 0.24 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0224654, 0.0246039, -0.0264032, 0.0510042, -0.0734695, 0.0510071
1: -0.0319891, 0.0601651, -0.0573585, 0.1537825, -0.1857716, 0.1175236
2: -0.0606338, 0.0999277, -0.0757487, 0.1525275, -0.2131613, 0.1756765
3: -0.0394671, 0.0654090, -0.0655027, 0.2003427, -0.2398098, 0.1309117
4: -0.0660760, 0.1172167, -0.1182454, 0.1901293, -0.2562054, 0.2354621

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A2_A1

### Relational analysis result of NS_A2_A1_A1_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880030, upper bound: 0.0887923
time: 0.26 seconds

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A2_A2

### Relational analysis result of NS_A2_A1_A1_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880195, upper bound: 0.0889940
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0204322, 0.0194142, -0.0212064, 0.0369534, -0.0573856, 0.0406206
1: -0.0283589, 0.0475317, -0.0486956, 0.1207656, -0.1491245, 0.0962273
2: -0.0497279, 0.0810393, -0.0575006, 0.1206669, -0.1703948, 0.1385399
3: -0.0324287, 0.0497364, -0.0555025, 0.1539112, -0.1863399, 0.1052389
4: -0.0500239, 0.0922334, -0.0888171, 0.1507825, -0.2008064, 0.1810506

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A1_B2_B2_A1_B1_B1

### Relational analysis result of NS_A2_A1_A1_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877547, upper bound: 0.0887195
time: 0.24 seconds

## Relational analysis of NS_A2_A1_A1_B2_B2_A1_B1_B2

### Relational analysis result of NS_A2_A1_A1_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877547, upper bound: 0.0887309
time: 0.28 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0204322, 0.0194142, -0.0264861, 0.0489378, -0.0693700, 0.0459003
1: -0.0283589, 0.0475317, -0.0554537, 0.1483759, -0.1767348, 0.1029854
2: -0.0497279, 0.0810393, -0.0748862, 0.1475046, -0.1972325, 0.1559255
3: -0.0324287, 0.0497364, -0.0630130, 0.1933062, -0.2257349, 0.1127493
4: -0.0500239, 0.0922334, -0.1155809, 0.1824649, -0.2324888, 0.2078144

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A1_B2_B2_A1_B2_B1

### Relational analysis result of NS_A2_A1_A1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877547, upper bound: 0.0889656
time: 0.25 seconds

## Relational analysis of NS_A2_A1_A1_B2_B2_A1_B2_B2

### Relational analysis result of NS_A2_A1_A1_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877547, upper bound: 0.0889656
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0232749, 0.0259113, -0.0212064, 0.0369534, -0.0602283, 0.0471177
1: -0.0340355, 0.0645256, -0.0486956, 0.1207656, -0.1548011, 0.1132213
2: -0.0633950, 0.1038888, -0.0575006, 0.1206669, -0.1840619, 0.1613894
3: -0.0410979, 0.0714647, -0.0555025, 0.1539112, -0.1950091, 0.1269672
4: -0.0702281, 0.1223172, -0.0888171, 0.1507825, -0.2210106, 0.2111343

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A1_B2_B2_A2_B1_B1

### Relational analysis result of NS_A2_A1_A1_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877581, upper bound: 0.0887872
time: 0.26 seconds

## Relational analysis of NS_A2_A1_A1_B2_B2_A2_B1_B2

### Relational analysis result of NS_A2_A1_A1_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877581, upper bound: 0.0888611
time: 0.24 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0232749, 0.0259113, -0.0264861, 0.0489378, -0.0722127, 0.0523974
1: -0.0340355, 0.0645256, -0.0554537, 0.1483759, -0.1824115, 0.1199793
2: -0.0633950, 0.1038888, -0.0748862, 0.1475046, -0.2108996, 0.1787751
3: -0.0410979, 0.0714647, -0.0630130, 0.1933062, -0.2344041, 0.1344776
4: -0.0702281, 0.1223172, -0.1155809, 0.1824649, -0.2526930, 0.2378981

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A1_B2_B2_A2_B2_B1

### Relational analysis result of NS_A2_A1_A1_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877581, upper bound: 0.0887872
time: 0.26 seconds

## Relational analysis of NS_A2_A1_A1_B2_B2_A2_B2_B2

### Relational analysis result of NS_A2_A1_A1_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877581, upper bound: 0.0888611
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0261405, 0.0324750, -0.0221645, 0.0274029, -0.0535434, 0.0546395
1: -0.0392807, 0.0896990, -0.0345445, 0.0750418, -0.1143225, 0.1242435
2: -0.0727650, 0.1226256, -0.0594218, 0.1060993, -0.1788643, 0.1820473
3: -0.0478143, 0.1099838, -0.0427037, 0.0898498, -0.1376641, 0.1526875
4: -0.0895612, 0.1459680, -0.0729370, 0.1265663, -0.2161275, 0.2189050

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A2_B1_B1_B1_A1_A1

### Relational analysis result of NS_A2_A1_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894647, upper bound: 0.0894345
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A2_B1_B1_B1_A1_A2

### Relational analysis result of NS_A2_A1_A2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894794, upper bound: 0.0893945
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0312804, 0.0368311, -0.0221645, 0.0274029, -0.0586833, 0.0589957
1: -0.0436182, 0.0962436, -0.0345445, 0.0750418, -0.1186600, 0.1307881
2: -0.0892005, 0.1371617, -0.0594218, 0.1060993, -0.1952998, 0.1965835
3: -0.0514396, 0.1168116, -0.0427037, 0.0898498, -0.1412894, 0.1595153
4: -0.1041426, 0.1617303, -0.0729370, 0.1265663, -0.2307089, 0.2346673

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A2_B1_B1_B1_A2_A1

### Relational analysis result of NS_A2_A1_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894647, upper bound: 0.0894345
time: 0.25 seconds

## Relational analysis of NS_A2_A1_A2_B1_B1_B1_A2_A2

### Relational analysis result of NS_A2_A1_A2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894794, upper bound: 0.0893945
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0261405, 0.0324750, -0.0278082, 0.0306353, -0.0567759, 0.0602832
1: -0.0392807, 0.0896990, -0.0394648, 0.0807291, -0.1200098, 0.1291638
2: -0.0727650, 0.1226256, -0.0763568, 0.1193576, -0.1921226, 0.1989824
3: -0.0478143, 0.1099838, -0.0468939, 0.0951927, -0.1430070, 0.1568777
4: -0.0895612, 0.1459680, -0.0870710, 0.1410980, -0.2306592, 0.2330390

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A2_B1_B1_B2_A1_A1

### Relational analysis result of NS_A2_A1_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894647, upper bound: 0.0894338
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A2_B1_B1_B2_A1_A2

### Relational analysis result of NS_A2_A1_A2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894794, upper bound: 0.0894107
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0312804, 0.0368311, -0.0278082, 0.0306353, -0.0619157, 0.0646393
1: -0.0436182, 0.0962436, -0.0394648, 0.0807291, -0.1243473, 0.1357084
2: -0.0892005, 0.1371617, -0.0763568, 0.1193576, -0.2085581, 0.2135185
3: -0.0514396, 0.1168116, -0.0468939, 0.0951927, -0.1466323, 0.1637055
4: -0.1041426, 0.1617303, -0.0870710, 0.1410980, -0.2452406, 0.2488013

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A2_B1_B1_B2_A2_A1

### Relational analysis result of NS_A2_A1_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894647, upper bound: 0.0894338
time: 0.29 seconds

## Relational analysis of NS_A2_A1_A2_B1_B1_B2_A2_A2

### Relational analysis result of NS_A2_A1_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894794, upper bound: 0.0893945
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0261405, 0.0324750, -0.0268901, 0.0335713, -0.0597118, 0.0593651
1: -0.0392807, 0.0896990, -0.0407071, 0.0941096, -0.1333904, 0.1304061
2: -0.0727650, 0.1226256, -0.0752188, 0.1259669, -0.1987319, 0.1978444
3: -0.0478143, 0.1099838, -0.0493354, 0.1159112, -0.1637255, 0.1593192
4: -0.0895612, 0.1459680, -0.0929475, 0.1502719, -0.2398331, 0.2389155

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A2_B1_B2_A1_B1_A1

### Relational analysis result of NS_A2_A1_A2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887914, upper bound: 0.0894893
time: 0.24 seconds

## Relational analysis of NS_A2_A1_A2_B1_B2_A1_B1_A2

### Relational analysis result of NS_A2_A1_A2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894290, upper bound: 0.0894605
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0261405, 0.0324750, -0.0321185, 0.0395926, -0.0657331, 0.0645934
1: -0.0392807, 0.0896990, -0.0450869, 0.1022720, -0.1415528, 0.1347859
2: -0.0727650, 0.1226256, -0.0918327, 0.1422157, -0.2149806, 0.2144583
3: -0.0478143, 0.1099838, -0.0529834, 0.1246344, -0.1724488, 0.1629672
4: -0.0895612, 0.1459680, -0.1079243, 0.1679169, -0.2574781, 0.2538923

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A2_B1_B2_A1_B2_B1

### Relational analysis result of NS_A2_A1_A2_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894454, upper bound: 0.0894549
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A2_B1_B2_A1_B2_B2

### Relational analysis result of NS_A2_A1_A2_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894290, upper bound: 0.0894605
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0312804, 0.0368311, -0.0268901, 0.0335713, -0.0648517, 0.0637212
1: -0.0436182, 0.0962436, -0.0407071, 0.0941096, -0.1377278, 0.1369507
2: -0.0892005, 0.1371617, -0.0752188, 0.1259669, -0.2151674, 0.2123805
3: -0.0514396, 0.1168116, -0.0493354, 0.1159112, -0.1673508, 0.1661470
4: -0.1041426, 0.1617303, -0.0929475, 0.1502719, -0.2544145, 0.2546778

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A2_B1_B2_A2_B1_A1

### Relational analysis result of NS_A2_A1_A2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894192, upper bound: 0.0894321
time: 0.28 seconds

## Relational analysis of NS_A2_A1_A2_B1_B2_A2_B1_A2

### Relational analysis result of NS_A2_A1_A2_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894285, upper bound: 0.0893968
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0312804, 0.0368311, -0.0321185, 0.0395926, -0.0708730, 0.0689496
1: -0.0436182, 0.0962436, -0.0450869, 0.1022720, -0.1458903, 0.1413305
2: -0.0892005, 0.1371617, -0.0918327, 0.1422157, -0.2314162, 0.2289944
3: -0.0514396, 0.1168116, -0.0529834, 0.1246344, -0.1760740, 0.1697951
4: -0.1041426, 0.1617303, -0.1079243, 0.1679169, -0.2720595, 0.2696546

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A2_B1_B2_A2_B2_B1

### Relational analysis result of NS_A2_A1_A2_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894323, upper bound: 0.0894032
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A2_B1_B2_A2_B2_B2

### Relational analysis result of NS_A2_A1_A2_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894285, upper bound: 0.0893968
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0182477, 0.0201999, -0.0264032, 0.0510042, -0.0692519, 0.0466032
1: -0.0264918, 0.0531019, -0.0573585, 0.1537825, -0.1802742, 0.1104604
2: -0.0432623, 0.0820052, -0.0757487, 0.1525275, -0.1957897, 0.1577540
3: -0.0321659, 0.0594380, -0.0655027, 0.2003427, -0.2325086, 0.1249406
4: -0.0501070, 0.0942402, -0.1182454, 0.1901293, -0.2402363, 0.2124856

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of NS_A2_A1_A2_B2_B1_A1_A1_B1

### Relational analysis result of NS_A2_A1_A2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0872646, upper bound: 0.0887290
time: 0.26 seconds

## Relational analysis of NS_A2_A1_A2_B2_B1_A1_A1_B2

### Relational analysis result of NS_A2_A1_A2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0879841, upper bound: 0.0889495
time: 0.25 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0235916, 0.0236044, -0.0264032, 0.0510042, -0.0745958, 0.0500076
1: -0.0315012, 0.0595827, -0.0573585, 0.1537825, -0.1852837, 0.1169412
2: -0.0606682, 0.0956602, -0.0757487, 0.1525275, -0.2131957, 0.1714089
3: -0.0362288, 0.0665506, -0.0655027, 0.2003427, -0.2365715, 0.1320532
4: -0.0642549, 0.1092418, -0.1182454, 0.1901293, -0.2543843, 0.2274871

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of NS_A2_A1_A2_B2_B1_A1_A2_B1

### Relational analysis result of NS_A2_A1_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0872646, upper bound: 0.0887290
time: 0.24 seconds

## Relational analysis of NS_A2_A1_A2_B2_B1_A1_A2_B2

### Relational analysis result of NS_A2_A1_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0879841, upper bound: 0.0889495
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0244806, 0.0282566, -0.0264032, 0.0510042, -0.0754848, 0.0546599
1: -0.0357648, 0.0724695, -0.0573585, 0.1537825, -0.1895473, 0.1298280
2: -0.0665385, 0.1101822, -0.0757487, 0.1525275, -0.2190659, 0.1859309
3: -0.0428190, 0.0833569, -0.0655027, 0.2003427, -0.2431617, 0.1488596
4: -0.0766040, 0.1293779, -0.1182454, 0.1901293, -0.2667333, 0.2476233

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of NS_A2_A1_A2_B2_B1_A2_A1_B1

### Relational analysis result of NS_A2_A1_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0872740, upper bound: 0.0889829
time: 0.26 seconds

## Relational analysis of NS_A2_A1_A2_B2_B1_A2_A1_B2

### Relational analysis result of NS_A2_A1_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0879935, upper bound: 0.0892049
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0278703, 0.0309839, -0.0264032, 0.0510042, -0.0788745, 0.0573872
1: -0.0392040, 0.0773109, -0.0573585, 0.1537825, -0.1929865, 0.1346694
2: -0.0779775, 0.1202548, -0.0757487, 0.1525275, -0.2305050, 0.1960035
3: -0.0459050, 0.0891398, -0.0655027, 0.2003427, -0.2462478, 0.1546424
4: -0.0874094, 0.1407572, -0.1182454, 0.1901293, -0.2775387, 0.2590026

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of NS_A2_A1_A2_B2_B1_A2_A2_B1

### Relational analysis result of NS_A2_A1_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0872740, upper bound: 0.0889829
time: 0.28 seconds

## Relational analysis of NS_A2_A1_A2_B2_B1_A2_A2_B2

### Relational analysis result of NS_A2_A1_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0879935, upper bound: 0.0892049
time: 0.24 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0248626, 0.0250574, -0.0212064, 0.0369534, -0.0618160, 0.0462638
1: -0.0340729, 0.0640337, -0.0486956, 0.1207656, -0.1548384, 0.1127293
2: -0.0645417, 0.1001453, -0.0575006, 0.1206669, -0.1852086, 0.1576459
3: -0.0381859, 0.0723889, -0.0555025, 0.1539112, -0.1920971, 0.1278914
4: -0.0690484, 0.1149325, -0.0888171, 0.1507825, -0.2198309, 0.2037497

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A2_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A2_B2_B2_A1_B1_B1

### Relational analysis result of NS_A2_A1_A2_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877666, upper bound: 0.0887447
time: 0.29 seconds

## Relational analysis of NS_A2_A1_A2_B2_B2_A1_B1_B2

### Relational analysis result of NS_A2_A1_A2_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877666, upper bound: 0.0887592
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0248626, 0.0250574, -0.0264861, 0.0489378, -0.0738004, 0.0515435
1: -0.0340729, 0.0640337, -0.0554537, 0.1483759, -0.1824488, 0.1194874
2: -0.0645417, 0.1001453, -0.0748862, 0.1475046, -0.2120463, 0.1750315
3: -0.0381859, 0.0723889, -0.0630130, 0.1933062, -0.2314921, 0.1354018
4: -0.0690484, 0.1149325, -0.1155809, 0.1824649, -0.2515133, 0.2305135

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A2_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A2_B2_B2_A1_B2_B1

### Relational analysis result of NS_A2_A1_A2_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877666, upper bound: 0.0888132
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A2_B2_B2_A1_B2_B2

### Relational analysis result of NS_A2_A1_A2_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877666, upper bound: 0.0887447
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0287746, 0.0323239, -0.0212064, 0.0369534, -0.0657280, 0.0535304
1: -0.0413172, 0.0814996, -0.0486956, 0.1207656, -0.1620828, 0.1301952
2: -0.0808285, 0.1242037, -0.0575006, 0.1206669, -0.2014954, 0.1817043
3: -0.0476249, 0.0947150, -0.0555025, 0.1539112, -0.2015361, 0.1502175
4: -0.0914771, 0.1458504, -0.0888171, 0.1507825, -0.2422595, 0.2346675

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A2_B2_B2_A2_B1_B1

### Relational analysis result of NS_A2_A1_A2_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877760, upper bound: 0.0888521
time: 0.25 seconds

## Relational analysis of NS_A2_A1_A2_B2_B2_A2_B1_B2

### Relational analysis result of NS_A2_A1_A2_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877760, upper bound: 0.0889126
time: 0.29 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0287746, 0.0323239, -0.0264861, 0.0489378, -0.0777124, 0.0588101
1: -0.0413172, 0.0814996, -0.0554537, 0.1483759, -0.1896931, 0.1369533
2: -0.0808285, 0.1242037, -0.0748862, 0.1475046, -0.2283331, 0.1990899
3: -0.0476249, 0.0947150, -0.0630130, 0.1933062, -0.2409311, 0.1577280
4: -0.0914771, 0.1458504, -0.1155809, 0.1824649, -0.2739420, 0.2614313

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_A1_A2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A2_B2_B2_A2_B2_B1

### Relational analysis result of NS_A2_A1_A2_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877760, upper bound: 0.0890443
time: 0.29 seconds

## Relational analysis of NS_A2_A1_A2_B2_B2_A2_B2_B2

### Relational analysis result of NS_A2_A1_A2_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877760, upper bound: 0.0890955
time: 0.27 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 1.64 seconds
NS_A1_B2_A1_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0885484, upper bound: 0.0888717
NS_A1_B2_A1_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0884127, upper bound: 0.0884127
NS_A1_B2_A1_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0885484, upper bound: 0.0890128
NS_A1_B2_A1_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0885484, upper bound: 0.0890429
NS_A1_B2_A1_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0883016, upper bound: 0.0888076
NS_A1_B2_A1_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0883016, upper bound: 0.0888377
NS_A1_B2_A1_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0883016, upper bound: 0.0889649
NS_A1_B2_A1_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0883016, upper bound: 0.0889950
NS_A1_B2_A1_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0884123, upper bound: 0.0885925
NS_A1_B2_A1_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0885889, upper bound: 0.0891682
NS_A1_B2_A1_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0885889, upper bound: 0.0891682
NS_A1_B2_A1_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0885889, upper bound: 0.0891682
NS_A1_B2_A1_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0883016, upper bound: 0.0889706
NS_A1_B2_A1_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0883016, upper bound: 0.0889951
NS_A1_B2_A1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0883016, upper bound: 0.0889649
NS_A1_B2_A1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0883016, upper bound: 0.0889950
NS_A1_B2_A1_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0884765, upper bound: 0.0891797
NS_A1_B2_A1_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0884765, upper bound: 0.0891797
NS_A1_B2_A1_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0884765, upper bound: 0.0891797
NS_A1_B2_A1_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0884765, upper bound: 0.0891797
NS_A1_B2_A1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0883148, upper bound: 0.0889823
NS_A1_B2_A1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0881659, upper bound: 0.0890131
NS_A1_B2_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0881659, upper bound: 0.0885881
NS_A1_B2_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0883148, upper bound: 0.0890131
NS_A1_B2_A1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0883057, upper bound: 0.0889817
NS_A1_B2_A1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0883057, upper bound: 0.0890131
NS_A1_B2_A1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0883057, upper bound: 0.0889817
NS_A1_B2_A1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0883057, upper bound: 0.0890131
NS_A2_A1_A1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894541, upper bound: 0.0894492
NS_A2_A1_A1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894454, upper bound: 0.0894453
NS_A2_A1_A1_B1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894453, upper bound: 0.0894573
NS_A2_A1_A1_B1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894453, upper bound: 0.0894615
NS_A2_A1_A1_B1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894345, upper bound: 0.0894647
NS_A2_A1_A1_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0893945, upper bound: 0.0894794
NS_A2_A1_A1_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894345, upper bound: 0.0894647
NS_A2_A1_A1_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0893945, upper bound: 0.0894794
NS_A2_A1_A1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0886072, upper bound: 0.0894442
NS_A2_A1_A1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894615, upper bound: 0.0894453
NS_A2_A1_A1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894573, upper bound: 0.0894442
NS_A2_A1_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894615, upper bound: 0.0894487
NS_A2_A1_A1_B1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894338, upper bound: 0.0894647
NS_A2_A1_A1_B1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894107, upper bound: 0.0894794
NS_A2_A1_A1_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0885372, upper bound: 0.0894647
NS_A2_A1_A1_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894107, upper bound: 0.0894794
NS_A2_A1_A1_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0880282, upper bound: 0.0885835
NS_A2_A1_A1_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0880235, upper bound: 0.0885306
NS_A2_A1_A1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0880282, upper bound: 0.0885835
NS_A2_A1_A1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0880235, upper bound: 0.0885306
NS_A2_A1_A1_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0880030, upper bound: 0.0887923
NS_A2_A1_A1_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0880195, upper bound: 0.0889269
NS_A2_A1_A1_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0880030, upper bound: 0.0887923
NS_A2_A1_A1_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0880195, upper bound: 0.0889940
NS_A2_A1_A1_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0877547, upper bound: 0.0887195
NS_A2_A1_A1_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0877547, upper bound: 0.0887309
NS_A2_A1_A1_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0877547, upper bound: 0.0889656
NS_A2_A1_A1_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0877547, upper bound: 0.0889656
NS_A2_A1_A1_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0877581, upper bound: 0.0887872
NS_A2_A1_A1_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0877581, upper bound: 0.0888611
NS_A2_A1_A1_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0877581, upper bound: 0.0887872
NS_A2_A1_A1_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0877581, upper bound: 0.0888611
NS_A2_A1_A2_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894647, upper bound: 0.0894345
NS_A2_A1_A2_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894794, upper bound: 0.0893945
NS_A2_A1_A2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894647, upper bound: 0.0894345
NS_A2_A1_A2_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894794, upper bound: 0.0893945
NS_A2_A1_A2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894647, upper bound: 0.0894338
NS_A2_A1_A2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894794, upper bound: 0.0894107
NS_A2_A1_A2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894647, upper bound: 0.0894338
NS_A2_A1_A2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894794, upper bound: 0.0893945
NS_A2_A1_A2_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0887914, upper bound: 0.0894893
NS_A2_A1_A2_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894290, upper bound: 0.0894605
NS_A2_A1_A2_B1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894454, upper bound: 0.0894549
NS_A2_A1_A2_B1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894290, upper bound: 0.0894605
NS_A2_A1_A2_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894192, upper bound: 0.0894321
NS_A2_A1_A2_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894285, upper bound: 0.0893968
NS_A2_A1_A2_B1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894323, upper bound: 0.0894032
NS_A2_A1_A2_B1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0894285, upper bound: 0.0893968
NS_A2_A1_A2_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0872646, upper bound: 0.0887290
NS_A2_A1_A2_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0879841, upper bound: 0.0889495
NS_A2_A1_A2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0872646, upper bound: 0.0887290
NS_A2_A1_A2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0879841, upper bound: 0.0889495
NS_A2_A1_A2_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0872740, upper bound: 0.0889829
NS_A2_A1_A2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0879935, upper bound: 0.0892049
NS_A2_A1_A2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0872740, upper bound: 0.0889829
NS_A2_A1_A2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0879935, upper bound: 0.0892049
NS_A2_A1_A2_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0877666, upper bound: 0.0887447
NS_A2_A1_A2_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0877666, upper bound: 0.0887592
NS_A2_A1_A2_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0877666, upper bound: 0.0888132
NS_A2_A1_A2_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0877666, upper bound: 0.0887447
NS_A2_A1_A2_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0877760, upper bound: 0.0888521
NS_A2_A1_A2_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0877760, upper bound: 0.0889126
NS_A2_A1_A2_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0877760, upper bound: 0.0890443
NS_A2_A1_A2_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 1.64
Output dim: 0, lower bound: -0.0877760, upper bound: 0.0890955

## BFS NS instance: NS_A1_B2_A1_B1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0183061, 0.0140802, -0.0156230, 0.0174385, -0.0357446, 0.0297032
1: -0.0240727, 0.0309017, -0.0229710, 0.0462277, -0.0703004, 0.0538727
2: -0.0450367, 0.0758257, -0.0355238, 0.0735236, -0.1185602, 0.1113495
3: -0.0263939, 0.0274213, -0.0294663, 0.0504758, -0.0768696, 0.0568875
4: -0.0390864, 0.0815583, -0.0411858, 0.0846141, -0.1237006, 0.1227442

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0183061, 0.0140802, -0.0227276, 0.0255313, -0.0438374, 0.0368078
1: -0.0240727, 0.0309017, -0.0326052, 0.0651657, -0.0892384, 0.0635068
2: -0.0450367, 0.0758257, -0.0614073, 0.1029931, -0.1480298, 0.1372330
3: -0.0263939, 0.0274213, -0.0403240, 0.0732887, -0.0996826, 0.0677453
4: -0.0390864, 0.0815583, -0.0688172, 0.1210865, -0.1601730, 0.1503756

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0235625, 0.0223077, -0.0156230, 0.0174385, -0.0410011, 0.0379307
1: -0.0298257, 0.0489782, -0.0229710, 0.0462277, -0.0760535, 0.0719492
2: -0.0665481, 0.0995724, -0.0355238, 0.0735236, -0.1400716, 0.1350962
3: -0.0366166, 0.0498813, -0.0294663, 0.0504758, -0.0870924, 0.0793476
4: -0.0645988, 0.1130393, -0.0411858, 0.0846141, -0.1492129, 0.1542251

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0235625, 0.0223077, -0.0227276, 0.0255313, -0.0490938, 0.0450353
1: -0.0298257, 0.0489782, -0.0326052, 0.0651657, -0.0949915, 0.0815833
2: -0.0665481, 0.0995724, -0.0614073, 0.1029931, -0.1695412, 0.1609797
3: -0.0366166, 0.0498813, -0.0403240, 0.0732887, -0.1099053, 0.0902054
4: -0.0645988, 0.1130393, -0.0688172, 0.1210865, -0.1856853, 0.1818565

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0182536, 0.0148930, -0.0156230, 0.0174385, -0.0356921, 0.0305160
1: -0.0242180, 0.0328669, -0.0229710, 0.0462277, -0.0704457, 0.0558379
2: -0.0444218, 0.0763963, -0.0355238, 0.0735236, -0.1179454, 0.1119201
3: -0.0269134, 0.0300028, -0.0294663, 0.0504758, -0.0773892, 0.0594691
4: -0.0399020, 0.0830934, -0.0411858, 0.0846141, -0.1245161, 0.1242793

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0182536, 0.0148930, -0.0227276, 0.0255313, -0.0437849, 0.0376206
1: -0.0242180, 0.0328669, -0.0326052, 0.0651657, -0.0893837, 0.0654721
2: -0.0444218, 0.0763963, -0.0614073, 0.1029931, -0.1474149, 0.1378036
3: -0.0269134, 0.0300028, -0.0403240, 0.0732887, -0.1002021, 0.0703268
4: -0.0399020, 0.0830934, -0.0688172, 0.1210865, -0.1609885, 0.1519107

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0268614, 0.0244703, -0.0156230, 0.0174385, -0.0443000, 0.0400934
1: -0.0328907, 0.0524811, -0.0229710, 0.0462277, -0.0791184, 0.0754521
2: -0.0769134, 0.1082844, -0.0355238, 0.0735236, -0.1504370, 0.1438082
3: -0.0390857, 0.0535550, -0.0294663, 0.0504758, -0.0895615, 0.0830212
4: -0.0741939, 0.1224238, -0.0411858, 0.0846141, -0.1588080, 0.1636096

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0268614, 0.0244703, -0.0227276, 0.0255313, -0.0523927, 0.0471980
1: -0.0328907, 0.0524811, -0.0326052, 0.0651657, -0.0980564, 0.0850863
2: -0.0769134, 0.1082844, -0.0614073, 0.1029931, -0.1799065, 0.1696917
3: -0.0390857, 0.0535550, -0.0403240, 0.0732887, -0.1123744, 0.0938790
4: -0.0741939, 0.1224238, -0.0688172, 0.1210865, -0.1952804, 0.1912410

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B1_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0183061, 0.0140802, -0.0211572, 0.0205201, -0.0388262, 0.0352374
1: -0.0240727, 0.0309017, -0.0285603, 0.0514196, -0.0754923, 0.0594620
2: -0.0450367, 0.0758257, -0.0523254, 0.0858022, -0.1308389, 0.1281511
3: -0.0263939, 0.0274213, -0.0333181, 0.0553236, -0.0817174, 0.0607393
4: -0.0390864, 0.0815583, -0.0536870, 0.0979762, -0.1370627, 0.1352453

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B1_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0183061, 0.0140802, -0.0261090, 0.0280248, -0.0463309, 0.0401892
1: -0.0240727, 0.0309017, -0.0360669, 0.0696426, -0.0937153, 0.0669686
2: -0.0450367, 0.0758257, -0.0719964, 0.1119072, -0.1569439, 0.1478221
3: -0.0263939, 0.0274213, -0.0433369, 0.0781435, -0.1045373, 0.0707582
4: -0.0390864, 0.0815583, -0.0783989, 0.1311053, -0.1701917, 0.1599573

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B1_B2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0182536, 0.0148930, -0.0211572, 0.0205201, -0.0387737, 0.0360502
1: -0.0242180, 0.0328669, -0.0285603, 0.0514196, -0.0756376, 0.0614272
2: -0.0444218, 0.0763963, -0.0523254, 0.0858022, -0.1302241, 0.1287217
3: -0.0269134, 0.0300028, -0.0333181, 0.0553236, -0.0822370, 0.0633209
4: -0.0399020, 0.0830934, -0.0536870, 0.0979762, -0.1378782, 0.1367804

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B1_B2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0182536, 0.0148930, -0.0261090, 0.0280248, -0.0462784, 0.0410020
1: -0.0242180, 0.0328669, -0.0360669, 0.0696426, -0.0938606, 0.0689338
2: -0.0444218, 0.0763963, -0.0719964, 0.1119072, -0.1563290, 0.1483927
3: -0.0269134, 0.0300028, -0.0433369, 0.0781435, -0.1050569, 0.0733398
4: -0.0399020, 0.0830934, -0.0783989, 0.1311053, -0.1710073, 0.1614924

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0235625, 0.0223077, -0.0211572, 0.0205201, -0.0440826, 0.0434648
1: -0.0298257, 0.0489782, -0.0285603, 0.0514196, -0.0812454, 0.0775385
2: -0.0665481, 0.0995724, -0.0523254, 0.0858022, -0.1523503, 0.1518978
3: -0.0366166, 0.0498813, -0.0333181, 0.0553236, -0.0919401, 0.0831994
4: -0.0645988, 0.1130393, -0.0536870, 0.0979762, -0.1625750, 0.1667262

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0235625, 0.0223077, -0.0261090, 0.0280248, -0.0515874, 0.0484167
1: -0.0298257, 0.0489782, -0.0360669, 0.0696426, -0.0994684, 0.0850451
2: -0.0665481, 0.0995724, -0.0719964, 0.1119072, -0.1784552, 0.1715688
3: -0.0366166, 0.0498813, -0.0433369, 0.0781435, -0.1147601, 0.0932183
4: -0.0645988, 0.1130393, -0.0783989, 0.1311053, -0.1957041, 0.1914382

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0268614, 0.0244703, -0.0211572, 0.0205201, -0.0473815, 0.0456275
1: -0.0328907, 0.0524811, -0.0285603, 0.0514196, -0.0843103, 0.0810414
2: -0.0769134, 0.1082844, -0.0523254, 0.0858022, -0.1627156, 0.1606098
3: -0.0390857, 0.0535550, -0.0333181, 0.0553236, -0.0944093, 0.0868731
4: -0.0741939, 0.1224238, -0.0536870, 0.0979762, -0.1721701, 0.1761108

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0268614, 0.0244703, -0.0261090, 0.0280248, -0.0548863, 0.0505793
1: -0.0328907, 0.0524811, -0.0360669, 0.0696426, -0.1025333, 0.0885480
2: -0.0769134, 0.1082844, -0.0719964, 0.1119072, -0.1888206, 0.1802808
3: -0.0390857, 0.0535550, -0.0433369, 0.0781435, -0.1172292, 0.0968919
4: -0.0741939, 0.1224238, -0.0783989, 0.1311053, -0.2052991, 0.2008227

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0183061, 0.0140802, -0.0201559, 0.0230345, -0.0413406, 0.0342361
1: -0.0240727, 0.0309017, -0.0289948, 0.0625832, -0.0866559, 0.0598964
2: -0.0450367, 0.0758257, -0.0497380, 0.0919095, -0.1369462, 0.1255637
3: -0.0263939, 0.0274213, -0.0353579, 0.0726474, -0.0990413, 0.0627791
4: -0.0390864, 0.0815583, -0.0592999, 0.1069696, -0.1460561, 0.1408583

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0183061, 0.0140802, -0.0285770, 0.0325231, -0.0508292, 0.0426572
1: -0.0240727, 0.0309017, -0.0403926, 0.0825989, -0.1066716, 0.0712942
2: -0.0450367, 0.0758257, -0.0792481, 0.1239909, -0.1690276, 0.1550738
3: -0.0263939, 0.0274213, -0.0472590, 0.0965308, -0.1229247, 0.0746802
4: -0.0390864, 0.0815583, -0.0901249, 0.1454333, -0.1845198, 0.1716832

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0182536, 0.0148930, -0.0201559, 0.0230345, -0.0412881, 0.0350489
1: -0.0242180, 0.0328669, -0.0289948, 0.0625832, -0.0868012, 0.0618617
2: -0.0444218, 0.0763963, -0.0497380, 0.0919095, -0.1363314, 0.1261343
3: -0.0269134, 0.0300028, -0.0353579, 0.0726474, -0.0995609, 0.0653607
4: -0.0399020, 0.0830934, -0.0592999, 0.1069696, -0.1468716, 0.1423934

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0182536, 0.0148930, -0.0285770, 0.0325231, -0.0507767, 0.0434700
1: -0.0242180, 0.0328669, -0.0403926, 0.0825989, -0.1068169, 0.0732595
2: -0.0444218, 0.0763963, -0.0792481, 0.1239909, -0.1684127, 0.1556444
3: -0.0269134, 0.0300028, -0.0472590, 0.0965308, -0.1234442, 0.0772618
4: -0.0399020, 0.0830934, -0.0901249, 0.1454333, -0.1853353, 0.1732183

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B2_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0235625, 0.0223077, -0.0201559, 0.0230345, -0.0465970, 0.0424636
1: -0.0298257, 0.0489782, -0.0289948, 0.0625832, -0.0924090, 0.0779729
2: -0.0665481, 0.0995724, -0.0497380, 0.0919095, -0.1584576, 0.1493104
3: -0.0366166, 0.0498813, -0.0353579, 0.0726474, -0.1092640, 0.0852392
4: -0.0645988, 0.1130393, -0.0592999, 0.1069696, -0.1715684, 0.1723392

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B2_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0235625, 0.0223077, -0.0285770, 0.0325231, -0.0560857, 0.0508846
1: -0.0298257, 0.0489782, -0.0403926, 0.0825989, -0.1124247, 0.0893708
2: -0.0665481, 0.0995724, -0.0792481, 0.1239909, -0.1905389, 0.1788205
3: -0.0366166, 0.0498813, -0.0472590, 0.0965308, -0.1331474, 0.0971403
4: -0.0645988, 0.1130393, -0.0901249, 0.1454333, -0.2100321, 0.2031641

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0268614, 0.0244703, -0.0201559, 0.0230345, -0.0498959, 0.0446262
1: -0.0328907, 0.0524811, -0.0289948, 0.0625832, -0.0954739, 0.0814759
2: -0.0769134, 0.1082844, -0.0497380, 0.0919095, -0.1688229, 0.1580224
3: -0.0390857, 0.0535550, -0.0353579, 0.0726474, -0.1117332, 0.0889129
4: -0.0741939, 0.1224238, -0.0592999, 0.1069696, -0.1811635, 0.1817237

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0268614, 0.0244703, -0.0285770, 0.0325231, -0.0593845, 0.0530473
1: -0.0328907, 0.0524811, -0.0403926, 0.0825989, -0.1154896, 0.0928737
2: -0.0769134, 0.1082844, -0.0792481, 0.1239909, -0.2009043, 0.1875325
3: -0.0390857, 0.0535550, -0.0472590, 0.0965308, -0.1356165, 0.1008140
4: -0.0741939, 0.1224238, -0.0901249, 0.1454333, -0.2196272, 0.2125487

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0235625, 0.0223077, -0.0254340, 0.0263058, -0.0498683, 0.0477417
1: -0.0298257, 0.0489782, -0.0336134, 0.0685003, -0.0983261, 0.0825915
2: -0.0665481, 0.0995724, -0.0668550, 0.1052071, -0.1717552, 0.1664273
3: -0.0366166, 0.0498813, -0.0392151, 0.0791419, -0.1157585, 0.0890964
4: -0.0645988, 0.1130393, -0.0734634, 0.1209796, -0.1855784, 0.1865027

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0235625, 0.0223077, -0.0319843, 0.0386605, -0.0622230, 0.0542920
1: -0.0298257, 0.0489782, -0.0438460, 0.0911426, -0.1209683, 0.0928242
2: -0.0665481, 0.0995724, -0.0905300, 0.1369953, -0.2035434, 0.1901023
3: -0.0366166, 0.0498813, -0.0502524, 0.1067187, -0.1433353, 0.1001338
4: -0.0645988, 0.1130393, -0.1015165, 0.1598046, -0.2244034, 0.2145557

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0268614, 0.0244703, -0.0254340, 0.0263058, -0.0531672, 0.0499044
1: -0.0328907, 0.0524811, -0.0336134, 0.0685003, -0.1013910, 0.0860945
2: -0.0769134, 0.1082844, -0.0668550, 0.1052071, -0.1821205, 0.1751394
3: -0.0390857, 0.0535550, -0.0392151, 0.0791419, -0.1182276, 0.0927700
4: -0.0741939, 0.1224238, -0.0734634, 0.1209796, -0.1951734, 0.1958872

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0268614, 0.0244703, -0.0319843, 0.0386605, -0.0655219, 0.0564547
1: -0.0328907, 0.0524811, -0.0438460, 0.0911426, -0.1240333, 0.0963271
2: -0.0769134, 0.1082844, -0.0905300, 0.1369953, -0.2139087, 0.1988143
3: -0.0390857, 0.0535550, -0.0502524, 0.1067187, -0.1458045, 0.1038074
4: -0.0741939, 0.1224238, -0.1015165, 0.1598046, -0.2339985, 0.2239402

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

## BFS NS instance: NS_A2_A1_A1_B1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0139696, 0.0151481, -0.0221645, 0.0274029, -0.0413724, 0.0373126
1: -0.0210482, 0.0384102, -0.0345445, 0.0750418, -0.0960899, 0.0729547
2: -0.0298601, 0.0651156, -0.0594218, 0.1060993, -0.1359594, 0.1245373
3: -0.0267865, 0.0398051, -0.0427037, 0.0898498, -0.1166363, 0.0825087
4: -0.0337018, 0.0741010, -0.0729370, 0.1265663, -0.1602681, 0.1470379

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A1_B1_A1_B1_B1_A1_A1

### Relational analysis result of NS_A2_A1_A1_B1_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0892464, upper bound: 0.0894332
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A1_B1_A1_B1_B1_A1_A2

### Relational analysis result of NS_A2_A1_A1_B1_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894311, upper bound: 0.0894448
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0186310, 0.0217224, -0.0221645, 0.0274029, -0.0460339, 0.0438869
1: -0.0282086, 0.0548497, -0.0345445, 0.0750418, -0.1032504, 0.0893943
2: -0.0483498, 0.0892622, -0.0594218, 0.1060993, -0.1544491, 0.1486840
3: -0.0359630, 0.0596494, -0.0427037, 0.0898498, -0.1258127, 0.1023531
4: -0.0550212, 0.1051150, -0.0729370, 0.1265663, -0.1815875, 0.1780520

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A1_B1_A1_B1_B1_A2_A1

### Relational analysis result of NS_A2_A1_A1_B1_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0892381, upper bound: 0.0893901
time: 0.26 seconds

## Relational analysis of NS_A2_A1_A1_B1_A1_B1_B1_A2_A2

### Relational analysis result of NS_A2_A1_A1_B1_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894184, upper bound: 0.0894184
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0215165, 0.0264666, -0.0201453, 0.0191321, -0.0406486, 0.0466119
1: -0.0332614, 0.0721512, -0.0276252, 0.0466993, -0.0799606, 0.0997763
2: -0.0573547, 0.1032774, -0.0488847, 0.0804011, -0.1377558, 0.1521620
3: -0.0413794, 0.0860633, -0.0320107, 0.0487060, -0.0900853, 0.1180740
4: -0.0701147, 0.1229001, -0.0489603, 0.0914019, -0.1615166, 0.1718604

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A1_B1_A1_B1_B2_B1_A1

### Relational analysis result of NS_A2_A1_A1_B1_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889242, upper bound: 0.0889428
time: 0.26 seconds

## Relational analysis of NS_A2_A1_A1_B1_A1_B1_B2_B1_A2

### Relational analysis result of NS_A2_A1_A1_B1_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889242, upper bound: 0.0889428
time: 0.24 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0215165, 0.0264666, -0.0231926, 0.0257079, -0.0472244, 0.0496591
1: -0.0332614, 0.0721512, -0.0334153, 0.0632880, -0.0965494, 0.1055664
2: -0.0573547, 0.1032774, -0.0630290, 0.1034112, -0.1607659, 0.1663064
3: -0.0413794, 0.0860633, -0.0409920, 0.0695329, -0.1109123, 0.1270553
4: -0.0701147, 0.1229001, -0.0691897, 0.1218800, -0.1919947, 0.1920899

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A1_B1_A1_B1_B2_B2_A1

### Relational analysis result of NS_A2_A1_A1_B1_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894285, upper bound: 0.0889737
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A1_B1_A1_B1_B2_B2_A2

### Relational analysis result of NS_A2_A1_A1_B1_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894285, upper bound: 0.0889737
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0215165, 0.0264666, -0.0189420, 0.0212494, -0.0427660, 0.0454086
1: -0.0332614, 0.0721512, -0.0277276, 0.0565663, -0.0898277, 0.0998787
2: -0.0573547, 0.1032774, -0.0455879, 0.0855275, -0.1428822, 0.1488653
3: -0.0413794, 0.0860633, -0.0334377, 0.0641360, -0.1055153, 0.1195010
4: -0.0701147, 0.1229001, -0.0534296, 0.0987422, -0.1688569, 0.1763297

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A1_B1_A1_B2_B1_B1_A1

### Relational analysis result of NS_A2_A1_A1_B1_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889551, upper bound: 0.0889613
time: 0.28 seconds

## Relational analysis of NS_A2_A1_A1_B1_A1_B2_B1_B1_A2

### Relational analysis result of NS_A2_A1_A1_B1_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889551, upper bound: 0.0889613
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0215165, 0.0264666, -0.0252627, 0.0295514, -0.0510679, 0.0517292
1: -0.0332614, 0.0721512, -0.0372698, 0.0770971, -0.1103585, 0.1094209
2: -0.0573547, 0.1032774, -0.0693456, 0.1142741, -0.1716288, 0.1726230
3: -0.0413794, 0.0860633, -0.0446361, 0.0896859, -0.1310653, 0.1306994
4: -0.0701147, 0.1229001, -0.0804323, 0.1348111, -0.2049258, 0.2033324

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A1_B1_A1_B2_B1_B2_A1

### Relational analysis result of NS_A2_A1_A1_B1_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894256, upper bound: 0.0889920
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A1_B1_A1_B2_B1_B2_A2

### Relational analysis result of NS_A2_A1_A1_B1_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894256, upper bound: 0.0889920
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0215165, 0.0264666, -0.0242722, 0.0246219, -0.0461384, 0.0507388
1: -0.0332614, 0.0721512, -0.0325581, 0.0628961, -0.0961575, 0.1047092
2: -0.0573547, 0.1032774, -0.0629592, 0.0991316, -0.1564863, 0.1662366
3: -0.0413794, 0.0860633, -0.0374444, 0.0711459, -0.1125253, 0.1235076
4: -0.0701147, 0.1229001, -0.0676032, 0.1136101, -0.1837248, 0.1905033

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A1_B1_A1_B2_B2_B1_A1

### Relational analysis result of NS_A2_A1_A1_B1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889141, upper bound: 0.0889588
time: 0.25 seconds

## Relational analysis of NS_A2_A1_A1_B1_A1_B2_B2_B1_A2

### Relational analysis result of NS_A2_A1_A1_B1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889141, upper bound: 0.0889588
time: 0.25 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0215165, 0.0264666, -0.0287080, 0.0323394, -0.0538560, 0.0551746
1: -0.0332614, 0.0721512, -0.0406874, 0.0820993, -0.1153607, 0.1128386
2: -0.0573547, 0.1032774, -0.0809133, 0.1245885, -0.1819432, 0.1841906
3: -0.0413794, 0.0860633, -0.0477162, 0.0956511, -0.1370305, 0.1337795
4: -0.0701147, 0.1229001, -0.0914368, 0.1465214, -0.2166361, 0.2143369

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A1_B1_A1_B2_B2_B2_A1

### Relational analysis result of NS_A2_A1_A1_B1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893777, upper bound: 0.0889915
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A1_B1_A1_B2_B2_B2_A2

### Relational analysis result of NS_A2_A1_A1_B1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893777, upper bound: 0.0894794
time: 0.28 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0195566, 0.0182853, -0.0221645, 0.0274029, -0.0469594, 0.0404498
1: -0.0267812, 0.0438515, -0.0345445, 0.0750418, -0.1018230, 0.0783960
2: -0.0469788, 0.0774942, -0.0594218, 0.1060993, -0.1530781, 0.1369159
3: -0.0308810, 0.0449339, -0.0427037, 0.0898498, -0.1207308, 0.0876376
4: -0.0465213, 0.0876333, -0.0729370, 0.1265663, -0.1730876, 0.1605702

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A1_B1_A2_B1_B1_A1_A1

### Relational analysis result of NS_A2_A1_A1_B1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891994, upper bound: 0.0894223
time: 0.28 seconds

## Relational analysis of NS_A2_A1_A1_B1_A2_B1_B1_A1_A2

### Relational analysis result of NS_A2_A1_A1_B1_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894412, upper bound: 0.0894399
time: 0.28 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0224532, 0.0245773, -0.0221645, 0.0274029, -0.0498560, 0.0467418
1: -0.0319785, 0.0600762, -0.0345445, 0.0750418, -0.1070202, 0.0946207
2: -0.0605661, 0.0998277, -0.0594218, 0.1060993, -0.1666654, 0.1592494
3: -0.0394435, 0.0652745, -0.0427037, 0.0898498, -0.1292933, 0.1079782
4: -0.0659716, 0.1170958, -0.0729370, 0.1265663, -0.1925378, 0.1900328

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A1_B1_A2_B1_B1_A2_A1

### Relational analysis result of NS_A2_A1_A1_B1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0892005, upper bound: 0.0893758
time: 0.25 seconds

## Relational analysis of NS_A2_A1_A1_B1_A2_B1_B1_A2_A2

### Relational analysis result of NS_A2_A1_A1_B1_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894446, upper bound: 0.0894054
time: 0.28 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0195566, 0.0182853, -0.0278082, 0.0306353, -0.0501919, 0.0460935
1: -0.0267812, 0.0438515, -0.0394648, 0.0807291, -0.1075103, 0.0833163
2: -0.0469788, 0.0774942, -0.0763568, 0.1193576, -0.1663364, 0.1538510
3: -0.0308810, 0.0449339, -0.0468939, 0.0951927, -0.1260737, 0.0918278
4: -0.0465213, 0.0876333, -0.0870710, 0.1410980, -0.1876193, 0.1747042

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A1_B1_A2_B1_B2_A1_A1

### Relational analysis result of NS_A2_A1_A1_B1_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891861, upper bound: 0.0894223
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A1_B1_A2_B1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of NS_A2_A1_A1_B1_A2_B1_B2_A1_B1

### Relational analysis result of NS_A2_A1_A1_B1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0892003, upper bound: 0.0889040
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A1_B1_A2_B1_B2_A1_B2

### Relational analysis result of NS_A2_A1_A1_B1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891986, upper bound: 0.0889041
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0224532, 0.0245773, -0.0278082, 0.0306353, -0.0530885, 0.0523855
1: -0.0319785, 0.0600762, -0.0394648, 0.0807291, -0.1127076, 0.0995409
2: -0.0605661, 0.0998277, -0.0763568, 0.1193576, -0.1799237, 0.1761845
3: -0.0394435, 0.0652745, -0.0468939, 0.0951927, -0.1346362, 0.1121684
4: -0.0659716, 0.1170958, -0.0870710, 0.1410980, -0.2070695, 0.2041668

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A1_B1_A2_B1_B2_A2_A1

### Relational analysis result of NS_A2_A1_A1_B1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886349, upper bound: 0.0893805
time: 0.26 seconds

## Relational analysis of NS_A2_A1_A1_B1_A2_B1_B2_A2_A2

### Relational analysis result of NS_A2_A1_A1_B1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894316, upper bound: 0.0894076
time: 0.28 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0271535, 0.0296738, -0.0189420, 0.0212494, -0.0484029, 0.0486158
1: -0.0381406, 0.0776904, -0.0277276, 0.0565663, -0.0947069, 0.1054179
2: -0.0742795, 0.1164777, -0.0455879, 0.0855275, -0.1598071, 0.1620656
3: -0.0456244, 0.0911491, -0.0334377, 0.0641360, -0.1097603, 0.1245868
4: -0.0842188, 0.1373387, -0.0534296, 0.0987422, -0.1829610, 0.1907683

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A1_B1_A2_B2_B1_B1_A1

### Relational analysis result of NS_A2_A1_A1_B1_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889572, upper bound: 0.0889448
time: 0.29 seconds

## Relational analysis of NS_A2_A1_A1_B1_A2_B2_B1_B1_A2

### Relational analysis result of NS_A2_A1_A1_B1_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889572, upper bound: 0.0894491
time: 0.28 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0271535, 0.0296738, -0.0252627, 0.0295514, -0.0567049, 0.0549365
1: -0.0381406, 0.0776904, -0.0372698, 0.0770971, -0.1152377, 0.1149602
2: -0.0742795, 0.1164777, -0.0693456, 0.1142741, -0.1885537, 0.1858233
3: -0.0456244, 0.0911491, -0.0446361, 0.0896859, -0.1353103, 0.1357852
4: -0.0842188, 0.1373387, -0.0804323, 0.1348111, -0.2190300, 0.2177710

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A1_B1_A2_B2_B1_B2_A1

### Relational analysis result of NS_A2_A1_A1_B1_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894277, upper bound: 0.0889756
time: 0.28 seconds

## Relational analysis of NS_A2_A1_A1_B1_A2_B2_B1_B2_A2

### Relational analysis result of NS_A2_A1_A1_B1_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894277, upper bound: 0.0894799
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0271535, 0.0296738, -0.0242722, 0.0246219, -0.0517754, 0.0539461
1: -0.0381406, 0.0776904, -0.0325581, 0.0628961, -0.1010367, 0.1102484
2: -0.0742795, 0.1164777, -0.0629592, 0.0991316, -0.1734111, 0.1794369
3: -0.0456244, 0.0911491, -0.0374444, 0.0711459, -0.1167703, 0.1285934
4: -0.0842188, 0.1373387, -0.0676032, 0.1136101, -0.1978289, 0.2049419

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A1_B1_A2_B2_B2_B1_A1

### Relational analysis result of NS_A2_A1_A1_B1_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889162, upper bound: 0.0889423
time: 0.28 seconds

## Relational analysis of NS_A2_A1_A1_B1_A2_B2_B2_B1_A2

### Relational analysis result of NS_A2_A1_A1_B1_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889162, upper bound: 0.0889423
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0271535, 0.0296738, -0.0287080, 0.0323394, -0.0594929, 0.0583818
1: -0.0381406, 0.0776904, -0.0406874, 0.0820993, -0.1202400, 0.1183778
2: -0.0742795, 0.1164777, -0.0809133, 0.1245885, -0.1988681, 0.1973909
3: -0.0456244, 0.0911491, -0.0477162, 0.0956511, -0.1412755, 0.1388653
4: -0.0842188, 0.1373387, -0.0914368, 0.1465214, -0.2307402, 0.2287755

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A1_B1_A2_B2_B2_B2_A1

### Relational analysis result of NS_A2_A1_A1_B1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893798, upper bound: 0.0889751
time: 0.26 seconds

## Relational analysis of NS_A2_A1_A1_B1_A2_B2_B2_B2_A2

### Relational analysis result of NS_A2_A1_A1_B1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893798, upper bound: 0.0894794
time: 0.29 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0111665, 0.0128576, -0.0264032, 0.0510042, -0.0621707, 0.0392608
1: -0.0177881, 0.0317167, -0.0573585, 0.1537825, -0.1715706, 0.0890752
2: -0.0214964, 0.0565724, -0.0757487, 0.1525275, -0.1740238, 0.1323211
3: -0.0239065, 0.0315967, -0.0655027, 0.2003427, -0.2242492, 0.0970993
4: -0.0257087, 0.0638043, -0.1182454, 0.1901293, -0.2158380, 0.1820497

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 21

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 3, pos: 16

### Candidate
type: A, layer: 3, pos: 21

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

### Candidate
type: A, layer: 3, pos: 39

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A1_A1_B1

### Relational analysis result of NS_A2_A1_A1_B2_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880154, upper bound: 0.0886050
time: 0.25 seconds

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 35

### Candidate
type: B, layer: 3, pos: 21

## BFS NS instance: NS_A2_A1_A1_B2_B1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0098449, 0.0119825, -0.0264032, 0.0510042, -0.0608491, 0.0383857
1: -0.0236876, 0.0284322, -0.0573585, 0.1537825, -0.1774701, 0.0857906
2: -0.0192642, 0.0471199, -0.0757487, 0.1525275, -0.1717916, 0.1228686
3: -0.0255710, 0.0249932, -0.0655027, 0.2003427, -0.2259137, 0.0904958
4: -0.0237004, 0.0517361, -0.1182454, 0.1901293, -0.2138297, 0.1699815

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 3, pos: 16

### Candidate
type: A, layer: 3, pos: 21

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

### Candidate
type: A, layer: 3, pos: 39

### Candidate
type: B, layer: 3, pos: 21

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A1_A2_B1

### Relational analysis result of NS_A2_A1_A1_B2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880105, upper bound: 0.0885119
time: 0.25 seconds

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 35

## BFS NS instance: NS_A2_A1_A1_B2_B1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0165749, 0.0156931, -0.0264032, 0.0510042, -0.0675791, 0.0420963
1: -0.0234446, 0.0372776, -0.0573585, 0.1537825, -0.1772271, 0.0946361
2: -0.0376981, 0.0687622, -0.0757487, 0.1525275, -0.1902255, 0.1445109
3: -0.0279595, 0.0369215, -0.0655027, 0.2003427, -0.2283023, 0.1024241
4: -0.0383014, 0.0773151, -0.1182454, 0.1901293, -0.2284307, 0.1955605

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 21

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 3, pos: 16

### Candidate
type: A, layer: 3, pos: 21

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A2_A1_A1

### Relational analysis result of NS_A2_A1_A1_B2_B1_A1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0867334, upper bound: 0.0873297
time: 0.24 seconds

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A2_A1_A2

### Relational analysis result of NS_A2_A1_A1_B2_B1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877824, upper bound: 0.0885073
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0144679, 0.0138485, -0.0264032, 0.0510042, -0.0654721, 0.0402517
1: -0.0272310, 0.0315756, -0.0573585, 0.1537825, -0.1810135, 0.0889341
2: -0.0315202, 0.0601238, -0.0757487, 0.1525275, -0.1840476, 0.1358725
3: -0.0279425, 0.0270811, -0.0655027, 0.2003427, -0.2282853, 0.0925838
4: -0.0326552, 0.0666849, -0.1182454, 0.1901293, -0.2227845, 0.1849303

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 3, pos: 16

### Candidate
type: A, layer: 3, pos: 21

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A2_A2_A1

### Relational analysis result of NS_A2_A1_A1_B2_B1_A1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0876202, upper bound: 0.0879461
time: 0.23 seconds

## Relational analysis of NS_A2_A1_A1_B2_B1_A1_A2_A2_A2

### Relational analysis result of NS_A2_A1_A1_B2_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877822, upper bound: 0.0884196
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0143064, 0.0182354, -0.0264032, 0.0510042, -0.0653105, 0.0446387
1: -0.0235850, 0.0468687, -0.0573585, 0.1537825, -0.1773675, 0.1042271
2: -0.0353422, 0.0772823, -0.0757487, 0.1525275, -0.1878696, 0.1530310
3: -0.0318207, 0.0501253, -0.0655027, 0.2003427, -0.2321634, 0.1156279
4: -0.0423230, 0.0912731, -0.1182454, 0.1901293, -0.2324524, 0.2095185

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 21

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 3, pos: 16

### Candidate
type: A, layer: 3, pos: 21

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 39

### Candidate
type: B, layer: 3, pos: 31

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A1_A1_B1

### Relational analysis result of NS_A2_A1_A1_B2_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0879896, upper bound: 0.0887806
time: 0.26 seconds

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 35

### Candidate
type: B, layer: 3, pos: 21

## BFS NS instance: NS_A2_A1_A1_B2_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0136071, 0.0147147, -0.0264032, 0.0510042, -0.0646113, 0.0411179
1: -0.0241999, 0.0330714, -0.0573585, 0.1537825, -0.1779823, 0.0904299
2: -0.0320616, 0.0647987, -0.0757487, 0.1525275, -0.1845890, 0.1405474
3: -0.0285303, 0.0291438, -0.0655027, 0.2003427, -0.2288731, 0.0946464
4: -0.0333022, 0.0740033, -0.1182454, 0.1901293, -0.2234315, 0.1922487

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 3, pos: 16

### Candidate
type: A, layer: 3, pos: 21

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

### Candidate
type: A, layer: 3, pos: 39

### Candidate
type: B, layer: 3, pos: 21

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A1_A2_B1

### Relational analysis result of NS_A2_A1_A1_B2_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880060, upper bound: 0.0889174
time: 0.25 seconds

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 35

## BFS NS instance: NS_A2_A1_A1_B2_B1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0189740, 0.0212108, -0.0264032, 0.0510042, -0.0699781, 0.0476140
1: -0.0279672, 0.0522051, -0.0573585, 0.1537825, -0.1817497, 0.1095636
2: -0.0492149, 0.0890881, -0.0757487, 0.1525275, -0.2017423, 0.1648368
3: -0.0354558, 0.0556172, -0.0655027, 0.2003427, -0.2357986, 0.1211198
4: -0.0547653, 0.1043473, -0.1182454, 0.1901293, -0.2448946, 0.2225927

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 21

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 3, pos: 16

### Candidate
type: A, layer: 3, pos: 21

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 39

### Candidate
type: B, layer: 3, pos: 31

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A2_A1_B1

### Relational analysis result of NS_A2_A1_A1_B2_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0879896, upper bound: 0.0887806
time: 0.28 seconds

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 35

### Candidate
type: B, layer: 3, pos: 21

## BFS NS instance: NS_A2_A1_A1_B2_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0168926, 0.0167259, -0.0264032, 0.0510042, -0.0678968, 0.0431292
1: -0.0271750, 0.0370017, -0.0573585, 0.1537825, -0.1809575, 0.0943602
2: -0.0423221, 0.0738940, -0.0757487, 0.1525275, -0.1948496, 0.1496428
3: -0.0308253, 0.0336026, -0.0655027, 0.2003427, -0.2311680, 0.0991053
4: -0.0425116, 0.0843122, -0.1182454, 0.1901293, -0.2326409, 0.2025576

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 21

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 3, pos: 16

### Candidate
type: A, layer: 3, pos: 21

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A2_A2_A1

### Relational analysis result of NS_A2_A1_A1_B2_B1_A2_A2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0874401, upper bound: 0.0871891
time: 0.24 seconds

## Relational analysis of NS_A2_A1_A1_B2_B1_A2_A2_A2_A2

### Relational analysis result of NS_A2_A1_A1_B2_B1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877861, upper bound: 0.0889230
time: 0.25 seconds

## BFS NS instance: NS_A2_A1_A1_B2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0204322, 0.0194142, -0.0147084, 0.0258321, -0.0462643, 0.0341226
1: -0.0283589, 0.0475317, -0.0399056, 0.0882198, -0.1165787, 0.0874373
2: -0.0497279, 0.0810393, -0.0335654, 0.0882203, -0.1379482, 0.1146047
3: -0.0324287, 0.0497364, -0.0436906, 0.1084482, -0.1408769, 0.0934270
4: -0.0500239, 0.0922334, -0.0516980, 0.1087936, -0.1588176, 0.1439314

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A2_A1_A1_B2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0204322, 0.0194142, -0.0176983, 0.0302437, -0.0506759, 0.0371125
1: -0.0283589, 0.0475317, -0.0443866, 0.0973887, -0.1257476, 0.0919183
2: -0.0497279, 0.0810393, -0.0460781, 0.1028996, -0.1526275, 0.1271174
3: -0.0324287, 0.0497364, -0.0486872, 0.1212652, -0.1536939, 0.0984236
4: -0.0500239, 0.0922334, -0.0702797, 0.1280541, -0.1780780, 0.1625131

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A2_A1_A1_B2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0204322, 0.0194142, -0.0182470, 0.0332807, -0.0537129, 0.0376612
1: -0.0283589, 0.0475317, -0.0453333, 0.1109483, -0.1393072, 0.0928650
2: -0.0497279, 0.0810393, -0.0471279, 0.1101631, -0.1598910, 0.1281672
3: -0.0324287, 0.0497364, -0.0498105, 0.1423159, -0.1747446, 0.0995469
4: -0.0500239, 0.0922334, -0.0771240, 0.1351168, -0.1851408, 0.1693574

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A2_A1_A1_B2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0204322, 0.0194142, -0.0223318, 0.0392295, -0.0596617, 0.0417459
1: -0.0283589, 0.0475317, -0.0510595, 0.1230056, -0.1513645, 0.0985912
2: -0.0497279, 0.0810393, -0.0630761, 0.1281110, -0.1778390, 0.1441154
3: -0.0324287, 0.0497364, -0.0567392, 0.1587957, -0.1912244, 0.1064756
4: -0.0500239, 0.0922334, -0.0985117, 0.1586076, -0.2086315, 0.1907451

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A2_A1_A1_B2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0232749, 0.0259113, -0.0147084, 0.0258321, -0.0491070, 0.0406196
1: -0.0340355, 0.0645256, -0.0399056, 0.0882198, -0.1222553, 0.1044312
2: -0.0633950, 0.1038888, -0.0335654, 0.0882203, -0.1516153, 0.1374542
3: -0.0410979, 0.0714647, -0.0436906, 0.1084482, -0.1495461, 0.1151553
4: -0.0702281, 0.1223172, -0.0516980, 0.1087936, -0.1790217, 0.1740152

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A2_A1_A1_B2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0232749, 0.0259113, -0.0176983, 0.0302437, -0.0535186, 0.0436096
1: -0.0340355, 0.0645256, -0.0443866, 0.0973887, -0.1314242, 0.1089122
2: -0.0633950, 0.1038888, -0.0460781, 0.1028996, -0.1662946, 0.1499670
3: -0.0410979, 0.0714647, -0.0486872, 0.1212652, -0.1623631, 0.1201519
4: -0.0702281, 0.1223172, -0.0702797, 0.1280541, -0.1982822, 0.1925969

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A2_A1_A1_B2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0232749, 0.0259113, -0.0182470, 0.0332807, -0.0565557, 0.0441583
1: -0.0340355, 0.0645256, -0.0453333, 0.1109483, -0.1449838, 0.1098590
2: -0.0633950, 0.1038888, -0.0471279, 0.1101631, -0.1735581, 0.1510168
3: -0.0410979, 0.0714647, -0.0498105, 0.1423159, -0.1834138, 0.1212752
4: -0.0702281, 0.1223172, -0.0771240, 0.1351168, -0.2053449, 0.1994411

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A2_A1_A1_B2_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0232749, 0.0259113, -0.0223318, 0.0392295, -0.0625044, 0.0482430
1: -0.0340355, 0.0645256, -0.0510595, 0.1230056, -0.1570411, 0.1155851
2: -0.0633950, 0.1038888, -0.0630761, 0.1281110, -0.1915060, 0.1669649
3: -0.0410979, 0.0714647, -0.0567392, 0.1587957, -0.1998935, 0.1282039
4: -0.0702281, 0.1223172, -0.0985117, 0.1586076, -0.2288357, 0.2208289

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A2_A1_A2_B1_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0182597, 0.0202359, -0.0221645, 0.0274029, -0.0456626, 0.0424004
1: -0.0265077, 0.0532302, -0.0345445, 0.0750418, -0.1015495, 0.0877747
2: -0.0433098, 0.0821251, -0.0594218, 0.1060993, -0.1494091, 0.1415469
3: -0.0321979, 0.0596275, -0.0427037, 0.0898498, -0.1220476, 0.1023312
4: -0.0502187, 0.0943898, -0.0729370, 0.1265663, -0.1767849, 0.1673267

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A2_B1_B1_B1_A1_A1_A1

### Relational analysis result of NS_A2_A1_A2_B1_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893852, upper bound: 0.0894854
time: 0.26 seconds

## Relational analysis of NS_A2_A1_A2_B1_B1_B1_A1_A1_A2

### Relational analysis result of NS_A2_A1_A2_B1_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894499, upper bound: 0.0894678
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0244806, 0.0282566, -0.0221645, 0.0274029, -0.0518835, 0.0504212
1: -0.0357648, 0.0724695, -0.0345445, 0.0750418, -0.1108066, 0.1070140
2: -0.0665385, 0.1101822, -0.0594218, 0.1060993, -0.1726378, 0.1696040
3: -0.0428190, 0.0833569, -0.0427037, 0.0898498, -0.1326687, 0.1260606
4: -0.0766040, 0.1293779, -0.0729370, 0.1265663, -0.2031703, 0.2023149

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A2_B1_B1_B1_A1_A2_A1

### Relational analysis result of NS_A2_A1_A2_B1_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893853, upper bound: 0.0894085
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A2_B1_B1_B1_A1_A2_A2

### Relational analysis result of NS_A2_A1_A2_B1_B1_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894575, upper bound: 0.0894160
time: 0.28 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0235916, 0.0236044, -0.0221645, 0.0274029, -0.0509945, 0.0457689
1: -0.0315012, 0.0595827, -0.0345445, 0.0750418, -0.1065430, 0.0941272
2: -0.0606682, 0.0956602, -0.0594218, 0.1060993, -0.1667675, 0.1550819
3: -0.0362288, 0.0665506, -0.0427037, 0.0898498, -0.1260786, 0.1092543
4: -0.0642549, 0.1092418, -0.0729370, 0.1265663, -0.1908212, 0.1821787

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A2_B1_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of NS_A2_A1_A2_B1_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of NS_A2_A1_A2_B1_B1_B1_A2_A1_B1

### Relational analysis result of NS_A2_A1_A2_B1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0892182, upper bound: 0.0890169
time: 0.28 seconds

## Relational analysis of NS_A2_A1_A2_B1_B1_B1_A2_A1_B2

### Relational analysis result of NS_A2_A1_A2_B1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0892150, upper bound: 0.0890169
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0278703, 0.0309839, -0.0221645, 0.0274029, -0.0552732, 0.0531485
1: -0.0392040, 0.0773109, -0.0345445, 0.0750418, -0.1142458, 0.1118554
2: -0.0779775, 0.1202548, -0.0594218, 0.1060993, -0.1840768, 0.1796765
3: -0.0459050, 0.0891398, -0.0427037, 0.0898498, -0.1357548, 0.1318434
4: -0.0874094, 0.1407572, -0.0729370, 0.1265663, -0.2139757, 0.2136942

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A2_B1_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of NS_A2_A1_A2_B1_B1_B1_A2_A2_B1

### Relational analysis result of NS_A2_A1_A2_B1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0892332, upper bound: 0.0891476
time: 0.28 seconds

## Relational analysis of NS_A2_A1_A2_B1_B1_B1_A2_A2_B2

### Relational analysis result of NS_A2_A1_A2_B1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0892244, upper bound: 0.0891641
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0182597, 0.0202359, -0.0278082, 0.0306353, -0.0488951, 0.0480441
1: -0.0265077, 0.0532302, -0.0394648, 0.0807291, -0.1072368, 0.0926950
2: -0.0433098, 0.0821251, -0.0763568, 0.1193576, -0.1626674, 0.1584819
3: -0.0321979, 0.0596275, -0.0468939, 0.0951927, -0.1273905, 0.1065214
4: -0.0502187, 0.0943898, -0.0870710, 0.1410980, -0.1913166, 0.1814607

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A2_B1_B1_B2_A1_A1_A1

### Relational analysis result of NS_A2_A1_A2_B1_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889401, upper bound: 0.0894854
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A2_B1_B1_B2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

### Candidate
type: B, layer: 3, pos: 24

## Relational analysis of NS_A2_A1_A2_B1_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of NS_A2_A1_A2_B1_B1_B2_A1_A1_B1

### Relational analysis result of NS_A2_A1_A2_B1_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0892086, upper bound: 0.0890336
time: 0.26 seconds

## Relational analysis of NS_A2_A1_A2_B1_B1_B2_A1_A1_B2

### Relational analysis result of NS_A2_A1_A2_B1_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0892105, upper bound: 0.0890337
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0244806, 0.0282566, -0.0278082, 0.0306353, -0.0551160, 0.0560648
1: -0.0357648, 0.0724695, -0.0394648, 0.0807291, -0.1164939, 0.1119343
2: -0.0665385, 0.1101822, -0.0763568, 0.1193576, -0.1858961, 0.1865390
3: -0.0428190, 0.0833569, -0.0468939, 0.0951927, -0.1380116, 0.1302508
4: -0.0766040, 0.1293779, -0.0870710, 0.1410980, -0.2177020, 0.2164489

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A2_B1_B1_B2_A1_A2_A1

### Relational analysis result of NS_A2_A1_A2_B1_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893723, upper bound: 0.0894346
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A2_B1_B1_B2_A1_A2_A2

### Relational analysis result of NS_A2_A1_A2_B1_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894445, upper bound: 0.0894422
time: 0.28 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0235916, 0.0236044, -0.0278082, 0.0306353, -0.0542269, 0.0514126
1: -0.0315012, 0.0595827, -0.0394648, 0.0807291, -0.1122303, 0.0990475
2: -0.0606682, 0.0956602, -0.0763568, 0.1193576, -0.1800259, 0.1720170
3: -0.0362288, 0.0665506, -0.0468939, 0.0951927, -0.1314215, 0.1134445
4: -0.0642549, 0.1092418, -0.0870710, 0.1410980, -0.2053529, 0.1963127

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A2_B1_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of NS_A2_A1_A2_B1_B1_B2_A2_A1_B1

### Relational analysis result of NS_A2_A1_A2_B1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0892074, upper bound: 0.0890167
time: 0.28 seconds

## Relational analysis of NS_A2_A1_A2_B1_B1_B2_A2_A1_B2

### Relational analysis result of NS_A2_A1_A2_B1_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0892105, upper bound: 0.0890167
time: 0.28 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0278703, 0.0309839, -0.0278082, 0.0306353, -0.0585056, 0.0587921
1: -0.0392040, 0.0773109, -0.0394648, 0.0807291, -0.1199331, 0.1167757
2: -0.0779775, 0.1202548, -0.0763568, 0.1193576, -0.1973352, 0.1966116
3: -0.0459050, 0.0891398, -0.0468939, 0.0951927, -0.1410977, 0.1360337
4: -0.0874094, 0.1407572, -0.0870710, 0.1410980, -0.2285074, 0.2278282

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A2_B1_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of NS_A2_A1_A2_B1_B1_B2_A2_A2_B1

### Relational analysis result of NS_A2_A1_A2_B1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0892224, upper bound: 0.0891476
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A2_B1_B1_B2_A2_A2_B2

### Relational analysis result of NS_A2_A1_A2_B1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0892199, upper bound: 0.0891641
time: 0.29 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0182597, 0.0202359, -0.0268901, 0.0335713, -0.0518310, 0.0471260
1: -0.0265077, 0.0532302, -0.0407071, 0.0941096, -0.1206174, 0.0939373
2: -0.0433098, 0.0821251, -0.0752188, 0.1259669, -0.1692767, 0.1573440
3: -0.0321979, 0.0596275, -0.0493354, 0.1159112, -0.1481090, 0.1089629
4: -0.0502187, 0.0943898, -0.0929475, 0.1502719, -0.2004906, 0.1873373

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A2_B1_B2_A1_B1_A1_A1

### Relational analysis result of NS_A2_A1_A2_B1_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893828, upper bound: 0.0894849
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A2_B1_B2_A1_B1_A1_A2

### Relational analysis result of NS_A2_A1_A2_B1_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894477, upper bound: 0.0894666
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0244806, 0.0282566, -0.0268901, 0.0335713, -0.0580519, 0.0551467
1: -0.0357648, 0.0724695, -0.0407071, 0.0941096, -0.1298744, 0.1131766
2: -0.0665385, 0.1101822, -0.0752188, 0.1259669, -0.1925054, 0.1854010
3: -0.0428190, 0.0833569, -0.0493354, 0.1159112, -0.1587301, 0.1326923
4: -0.0766040, 0.1293779, -0.0929475, 0.1502719, -0.2268759, 0.2223255

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 35

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A2_B1_B2_A1_B1_A2_A1

### Relational analysis result of NS_A2_A1_A2_B1_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893829, upper bound: 0.0894337
time: 0.28 seconds

## Relational analysis of NS_A2_A1_A2_B1_B2_A1_B1_A2_A2

### Relational analysis result of NS_A2_A1_A2_B1_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894550, upper bound: 0.0894314
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0261405, 0.0324750, -0.0242722, 0.0246219, -0.0507624, 0.0567472
1: -0.0392807, 0.0896990, -0.0325581, 0.0628961, -0.1021769, 0.1222571
2: -0.0727650, 0.1226256, -0.0629592, 0.0991316, -0.1718966, 0.1855848
3: -0.0478143, 0.1099838, -0.0374444, 0.0711459, -0.1189602, 0.1474282
4: -0.0895612, 0.1459680, -0.0676032, 0.1136101, -0.2031713, 0.2135712

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A2_B1_B2_A1_B2_B1_A1

### Relational analysis result of NS_A2_A1_A2_B1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889347, upper bound: 0.0889732
time: 0.29 seconds

## Relational analysis of NS_A2_A1_A2_B1_B2_A1_B2_B1_A2

### Relational analysis result of NS_A2_A1_A2_B1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889347, upper bound: 0.0889732
time: 0.28 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0261405, 0.0324750, -0.0287080, 0.0323394, -0.0584800, 0.0611830
1: -0.0392807, 0.0896990, -0.0406874, 0.0820993, -0.1213801, 0.1303864
2: -0.0727650, 0.1226256, -0.0809133, 0.1245885, -0.1973535, 0.2035388
3: -0.0478143, 0.1099838, -0.0477162, 0.0956511, -0.1434654, 0.1577000
4: -0.0895612, 0.1459680, -0.0914368, 0.1465214, -0.2360826, 0.2374048

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A2_B1_B2_A1_B2_B2_A1

### Relational analysis result of NS_A2_A1_A2_B1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893983, upper bound: 0.0890023
time: 0.29 seconds

## Relational analysis of NS_A2_A1_A2_B1_B2_A1_B2_B2_A2

### Relational analysis result of NS_A2_A1_A2_B1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893983, upper bound: 0.0890023
time: 0.30 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0235916, 0.0236044, -0.0268901, 0.0335713, -0.0571629, 0.0504945
1: -0.0315012, 0.0595827, -0.0407071, 0.0941096, -0.1256108, 0.1002898
2: -0.0606682, 0.0956602, -0.0752188, 0.1259669, -0.1866351, 0.1708790
3: -0.0362288, 0.0665506, -0.0493354, 0.1159112, -0.1521400, 0.1158859
4: -0.0642549, 0.1092418, -0.0929475, 0.1502719, -0.2145268, 0.2021893

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A2_B1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_A1_A2_B1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894387, upper bound: 0.0893278
time: 0.28 seconds

## Relational analysis of NS_A2_A1_A2_B1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_A1_A2_B1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894477, upper bound: 0.0894284
time: 0.28 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0278703, 0.0309839, -0.0268901, 0.0335713, -0.0614416, 0.0578740
1: -0.0392040, 0.0773109, -0.0407071, 0.0941096, -0.1333137, 0.1180180
2: -0.0779775, 0.1202548, -0.0752188, 0.1259669, -0.2039444, 0.1954736
3: -0.0459050, 0.0891398, -0.0493354, 0.1159112, -0.1618162, 0.1384751
4: -0.0874094, 0.1407572, -0.0929475, 0.1502719, -0.2376813, 0.2337048

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A2_B1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_A1_A2_B1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894476, upper bound: 0.0891839
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A2_B1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_A1_A2_B1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894552, upper bound: 0.0893635
time: 0.28 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0312804, 0.0368311, -0.0242722, 0.0246219, -0.0559023, 0.0611034
1: -0.0436182, 0.0962436, -0.0325581, 0.0628961, -0.1065143, 0.1288016
2: -0.0892005, 0.1371617, -0.0629592, 0.0991316, -0.1883321, 0.2001209
3: -0.0514396, 0.1168116, -0.0374444, 0.0711459, -0.1225855, 0.1542560
4: -0.1041426, 0.1617303, -0.0676032, 0.1136101, -0.2177527, 0.2293335

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A2_B1_B2_A2_B2_B1_A1

### Relational analysis result of NS_A2_A1_A2_B1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889322, upper bound: 0.0889157
time: 0.28 seconds

## Relational analysis of NS_A2_A1_A2_B1_B2_A2_B2_B1_A2

### Relational analysis result of NS_A2_A1_A2_B1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889322, upper bound: 0.0893777
time: 0.28 seconds

## BFS NS instance: NS_A2_A1_A2_B1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0312804, 0.0368311, -0.0287080, 0.0323394, -0.0636198, 0.0655391
1: -0.0436182, 0.0962436, -0.0406874, 0.0820993, -0.1257176, 0.1369310
2: -0.0892005, 0.1371617, -0.0809133, 0.1245885, -0.2137890, 0.2180750
3: -0.0514396, 0.1168116, -0.0477162, 0.0956511, -0.1470907, 0.1645278
4: -0.1041426, 0.1617303, -0.0914368, 0.1465214, -0.2506640, 0.2531671

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_A2_B1_B2_A2_B2_B2_A1

### Relational analysis result of NS_A2_A1_A2_B1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893958, upper bound: 0.0889459
time: 0.29 seconds

## Relational analysis of NS_A2_A1_A2_B1_B2_A2_B2_B2_A2

### Relational analysis result of NS_A2_A1_A2_B1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893958, upper bound: 0.0889459
time: 0.28 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0182477, 0.0201999, -0.0230866, 0.0221183, -0.0403660, 0.0432865
1: -0.0264918, 0.0531019, -0.0277468, 0.0522709, -0.0787626, 0.0808487
2: -0.0432623, 0.0820052, -0.0649820, 0.0940102, -0.1372724, 0.1469873
3: -0.0321659, 0.0594380, -0.0325028, 0.0593246, -0.0914905, 0.0919408
4: -0.0501070, 0.0942402, -0.0711249, 0.1008022, -0.1509092, 0.1653650

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of NS_A2_A1_A2_B2_B1_A1_A1_B1_A1

### Relational analysis result of NS_A2_A1_A2_B2_B1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0867634, upper bound: 0.0868607
time: 0.26 seconds

## Relational analysis of NS_A2_A1_A2_B2_B1_A1_A1_B1_A2

### Relational analysis result of NS_A2_A1_A2_B2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0867634, upper bound: 0.0887707
time: 0.28 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0182477, 0.0201999, -0.0260989, 0.0498248, -0.0680725, 0.0462988
1: -0.0264918, 0.0531019, -0.0566694, 0.1516231, -0.1781148, 0.1097714
2: -0.0432623, 0.0820052, -0.0746548, 0.1504116, -0.1936738, 0.1566601
3: -0.0321659, 0.0594380, -0.0647910, 0.1975340, -0.2296999, 0.1242290
4: -0.0501070, 0.0942402, -0.1166729, 0.1874111, -0.2375180, 0.2109131

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 24
type: B, layer: 3, pos: 24
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 16

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of NS_A2_A1_A2_B2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 24

## Relational analysis of NS_A2_A1_A2_B2_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 24

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of NS_A2_A1_A2_B2_B1_A1_A1_B2_A1

### Relational analysis result of NS_A2_A1_A2_B2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0879176, upper bound: 0.0886671
time: 0.26 seconds

## Relational analysis of NS_A2_A1_A2_B2_B1_A1_A1_B2_A2

### Relational analysis result of NS_A2_A1_A2_B2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0879176, upper bound: 0.0886424
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0235916, 0.0236044, -0.0230866, 0.0221183, -0.0457100, 0.0466910
1: -0.0315012, 0.0595827, -0.0277468, 0.0522709, -0.0837721, 0.0873295
2: -0.0606682, 0.0956602, -0.0649820, 0.0940102, -0.1546784, 0.1606422
3: -0.0362288, 0.0665506, -0.0325028, 0.0593246, -0.0955534, 0.0990534
4: -0.0642549, 0.1092418, -0.0711249, 0.1008022, -0.1650572, 0.1803666

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of NS_A2_A1_A2_B2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of NS_A2_A1_A2_B2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of NS_A2_A1_A2_B2_B1_A1_A2_B1_B1

### Relational analysis result of NS_A2_A1_A2_B2_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0866955, upper bound: 0.0884541
time: 0.25 seconds

## Relational analysis of NS_A2_A1_A2_B2_B1_A1_A2_B1_B2

### Relational analysis result of NS_A2_A1_A2_B2_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0869759, upper bound: 0.0885220
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0235916, 0.0236044, -0.0260989, 0.0498248, -0.0734165, 0.0497033
1: -0.0315012, 0.0595827, -0.0566694, 0.1516231, -0.1831243, 0.1162521
2: -0.0606682, 0.0956602, -0.0746548, 0.1504116, -0.2110798, 0.1703150
3: -0.0362288, 0.0665506, -0.0647910, 0.1975340, -0.2337628, 0.1313416
4: -0.0642549, 0.1092418, -0.1166729, 0.1874111, -0.2516660, 0.2259147

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 16

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of NS_A2_A1_A2_B2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of NS_A2_A1_A2_B2_B1_A1_A2_B2_A1

### Relational analysis result of NS_A2_A1_A2_B2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0879176, upper bound: 0.0885430
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A2_B2_B1_A1_A2_B2_A2

### Relational analysis result of NS_A2_A1_A2_B2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0879176, upper bound: 0.0885856
time: 0.25 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0244806, 0.0282566, -0.0230866, 0.0221183, -0.0465990, 0.0513432
1: -0.0357648, 0.0724695, -0.0277468, 0.0522709, -0.0880357, 0.1002163
2: -0.0665385, 0.1101822, -0.0649820, 0.0940102, -0.1605486, 0.1751642
3: -0.0428190, 0.0833569, -0.0325028, 0.0593246, -0.1021436, 0.1158597
4: -0.0766040, 0.1293779, -0.0711249, 0.1008022, -0.1774062, 0.2005028

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of NS_A2_A1_A2_B2_B1_A2_A1_B1_A1

### Relational analysis result of NS_A2_A1_A2_B2_B1_A2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0867340, upper bound: 0.0867036
time: 0.25 seconds

## Relational analysis of NS_A2_A1_A2_B2_B1_A2_A1_B1_A2

### Relational analysis result of NS_A2_A1_A2_B2_B1_A2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0867340, upper bound: 0.0867036
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0244806, 0.0282566, -0.0260989, 0.0498248, -0.0743055, 0.0543555
1: -0.0357648, 0.0724695, -0.0566694, 0.1516231, -0.1873879, 0.1291389
2: -0.0665385, 0.1101822, -0.0746548, 0.1504116, -0.2169500, 0.1848370
3: -0.0428190, 0.0833569, -0.0647910, 0.1975340, -0.2403530, 0.1481480
4: -0.0766040, 0.1293779, -0.1166729, 0.1874111, -0.2640151, 0.2460508

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 16

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of NS_A2_A1_A2_B2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of NS_A2_A1_A2_B2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of NS_A2_A1_A2_B2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of NS_A2_A1_A2_B2_B1_A2_A1_B2_A1

### Relational analysis result of NS_A2_A1_A2_B2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0879163, upper bound: 0.0888022
time: 0.27 seconds

## Relational analysis of NS_A2_A1_A2_B2_B1_A2_A1_B2_A2

### Relational analysis result of NS_A2_A1_A2_B2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0879274, upper bound: 0.0889834
time: 0.28 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0278703, 0.0309839, -0.0230866, 0.0221183, -0.0499886, 0.0540705
1: -0.0392040, 0.0773109, -0.0277468, 0.0522709, -0.0914749, 0.1050577
2: -0.0779775, 0.1202548, -0.0649820, 0.0940102, -0.1719877, 0.1852368
3: -0.0459050, 0.0891398, -0.0325028, 0.0593246, -0.1052297, 0.1216426
4: -0.0874094, 0.1407572, -0.0711249, 0.1008022, -0.1882116, 0.2118821

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 24

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of NS_A2_A1_A2_B2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of NS_A2_A1_A2_B2_B1_A2_A2_B1_B1

### Relational analysis result of NS_A2_A1_A2_B2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0867058, upper bound: 0.0886799
time: 0.28 seconds

## Relational analysis of NS_A2_A1_A2_B2_B1_A2_A2_B1_B2

### Relational analysis result of NS_A2_A1_A2_B2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0869862, upper bound: 0.0888516
time: 0.26 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0278703, 0.0309839, -0.0260989, 0.0498248, -0.0776951, 0.0570828
1: -0.0392040, 0.0773109, -0.0566694, 0.1516231, -0.1908271, 0.1339803
2: -0.0779775, 0.1202548, -0.0746548, 0.1504116, -0.2283891, 0.1949096
3: -0.0459050, 0.0891398, -0.0647910, 0.1975340, -0.2434391, 0.1539308
4: -0.0874094, 0.1407572, -0.1166729, 0.1874111, -0.2748204, 0.2574301

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 24
type: B, layer: 3, pos: 35
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 16

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of NS_A2_A1_A2_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of NS_A2_A1_A2_B2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 31

## Relational analysis of NS_A2_A1_A2_B2_B1_A2_A2_B2_A1

### Relational analysis result of NS_A2_A1_A2_B2_B1_A2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0873594, upper bound: 0.0872622
time: 0.28 seconds

## Relational analysis of NS_A2_A1_A2_B2_B1_A2_A2_B2_A2

### Relational analysis result of NS_A2_A1_A2_B2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0876867, upper bound: 0.0891036
time: 0.27 seconds

## BFS NS instance: NS_A2_A1_A2_B2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0248626, 0.0250574, -0.0147084, 0.0258321, -0.0506947, 0.0397657
1: -0.0340729, 0.0640337, -0.0399056, 0.0882198, -0.1222927, 0.1039392
2: -0.0645417, 0.1001453, -0.0335654, 0.0882203, -0.1527620, 0.1337107
3: -0.0381859, 0.0723889, -0.0436906, 0.1084482, -0.1466341, 0.1160795
4: -0.0690484, 0.1149325, -0.0516980, 0.1087936, -0.1778420, 0.1666305

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A2_A1_A2_B2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0248626, 0.0250574, -0.0176983, 0.0302437, -0.0551063, 0.0427557
1: -0.0340729, 0.0640337, -0.0443866, 0.0973887, -0.1314615, 0.1084202
2: -0.0645417, 0.1001453, -0.0460781, 0.1028996, -0.1674413, 0.1462234
3: -0.0381859, 0.0723889, -0.0486872, 0.1212652, -0.1594511, 0.1210761
4: -0.0690484, 0.1149325, -0.0702797, 0.1280541, -0.1971024, 0.1852122

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A2_A1_A2_B2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0248626, 0.0250574, -0.0182470, 0.0332807, -0.0581434, 0.0433044
1: -0.0340729, 0.0640337, -0.0453333, 0.1109483, -0.1450211, 0.1093670
2: -0.0645417, 0.1001453, -0.0471279, 0.1101631, -0.1747048, 0.1472732
3: -0.0381859, 0.0723889, -0.0498105, 0.1423159, -0.1805018, 0.1221994
4: -0.0690484, 0.1149325, -0.0771240, 0.1351168, -0.2041652, 0.1920565

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A2_A1_A2_B2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0248626, 0.0250574, -0.0223318, 0.0392295, -0.0640921, 0.0473891
1: -0.0340729, 0.0640337, -0.0510595, 0.1230056, -0.1570785, 0.1150932
2: -0.0645417, 0.1001453, -0.0630761, 0.1281110, -0.1926527, 0.1632214
3: -0.0381859, 0.0723889, -0.0567392, 0.1587957, -0.1969815, 0.1291281
4: -0.0690484, 0.1149325, -0.0985117, 0.1586076, -0.2276560, 0.2134442

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A2_A1_A2_B2_B2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0287746, 0.0323239, -0.0147084, 0.0258321, -0.0546067, 0.0470323
1: -0.0413172, 0.0814996, -0.0399056, 0.0882198, -0.1295370, 0.1214052
2: -0.0808285, 0.1242037, -0.0335654, 0.0882203, -0.1690488, 0.1577691
3: -0.0476249, 0.0947150, -0.0436906, 0.1084482, -0.1560731, 0.1384056
4: -0.0914771, 0.1458504, -0.0516980, 0.1087936, -0.2002707, 0.1975484

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A2_A1_A2_B2_B2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0287746, 0.0323239, -0.0176983, 0.0302437, -0.0590183, 0.0500223
1: -0.0413172, 0.0814996, -0.0443866, 0.0973887, -0.1387059, 0.1258862
2: -0.0808285, 0.1242037, -0.0460781, 0.1028996, -0.1837281, 0.1702818
3: -0.0476249, 0.0947150, -0.0486872, 0.1212652, -0.1688901, 0.1434022
4: -0.0914771, 0.1458504, -0.0702797, 0.1280541, -0.2195311, 0.2161301

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A2_A1_A2_B2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0287746, 0.0323239, -0.0182470, 0.0332807, -0.0620554, 0.0505710
1: -0.0413172, 0.0814996, -0.0453333, 0.1109483, -0.1522655, 0.1268329
2: -0.0808285, 0.1242037, -0.0471279, 0.1101631, -0.1909916, 0.1713316
3: -0.0476249, 0.0947150, -0.0498105, 0.1423159, -0.1899408, 0.1445255
4: -0.0914771, 0.1458504, -0.0771240, 0.1351168, -0.2265939, 0.2229744

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## BFS NS instance: NS_A2_A1_A2_B2_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0287746, 0.0323239, -0.0223318, 0.0392295, -0.0680041, 0.0546557
1: -0.0413172, 0.0814996, -0.0510595, 0.1230056, -0.1643228, 0.1325591
2: -0.0808285, 0.1242037, -0.0630761, 0.1281110, -0.2089395, 0.1872798
3: -0.0476249, 0.0947150, -0.0567392, 0.1587957, -0.2064205, 0.1514542
4: -0.0914771, 0.1458504, -0.0985117, 0.1586076, -0.2500847, 0.2443621

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 1.53 + 293.28 = 294.81 seconds
