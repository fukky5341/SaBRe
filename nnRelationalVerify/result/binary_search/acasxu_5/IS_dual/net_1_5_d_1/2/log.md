## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_5.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 0.088187946


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0390887, 0.0634215, -0.0390887, 0.0634215, -0.1025102, 0.1025102)
1: (-0.0557757, 0.1414140, -0.0557757, 0.1414140, -0.1971897, 0.1971898)
2: (-0.1134923, 0.1820314, -0.1134923, 0.1820314, -0.2955237, 0.2955237)
3: (-0.0636387, 0.1747649, -0.0636387, 0.1747649, -0.2384036, 0.2384035)
4: (-0.1363116, 0.2151742, -0.1363116, 0.2151742, -0.3514858, 0.3514858)

## BASE Result
execution time: IAR + LP analysis = 1.52 + 0.91 = 2.43 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0899877, upper bound: 0.0899877


# Binary Search by BASE starts (time budget: 1197.57 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1000000, mid=0.1000000, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 1) starts
Candidate diff: 0.0500000


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0500000, mid=0.0500000, abs_max=0.10251016169786453
rel_dist={0: [-0.0899835175585181, 0.08998351755851813]}

## Binary search (step 2) starts
Candidate diff: 0.0250000


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0250000, mid=0.0250000, abs_max=0.10251016169786453
rel_dist={0: [-0.08996129100877424, 0.08996129100877422]}

## Binary search (step 3) starts
Candidate diff: 0.0125000


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0125000, mid=0.0125000, abs_max=0.10251016169786453
rel_dist={0: [-0.0899476251703134, 0.08994762517031346]}

## Binary search (step 4) starts
Candidate diff: 0.0062500


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0062500, mid=0.0062500, abs_max=0.10251016169786453
rel_dist={0: [-0.08994001998180878, 0.08994001998180878]}

## Binary search (step 5) starts
Candidate diff: 0.0031250


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0031250, mid=0.0031250, abs_max=0.10251016169786453
rel_dist={0: [-0.08993124712088736, 0.08993124712088738]}

## Binary search (step 6) starts
Candidate diff: 0.0015625


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0015625, mid=0.0015625, abs_max=0.10251016169786453
rel_dist={0: [-0.08992489701688348, 0.08992489701674958]}

## Binary search (step 7) starts
Candidate diff: 0.0007812


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0007812, mid=0.0007812, abs_max=0.10251016169786453
rel_dist={0: [-0.08991947978566078, 0.08991947978566078]}

## Binary search (step 8) starts
Candidate diff: 0.0003906


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0003906, mid=0.0003906, abs_max=0.10251016169786453
rel_dist={0: [-0.0899161563368783, 0.08991615633673694]}

## Binary search (step 9) starts
Candidate diff: 0.0001953


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0001953, mid=0.0001953, abs_max=0.10251016169786453
rel_dist={0: [-0.0899141627247981, 0.08991416272479807]}

## Binary search (step 10) starts
Candidate diff: 0.0000977


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0000977, mid=0.0000977, abs_max=0.10251016169786453
rel_dist={0: [-0.08991292356463379, 0.08991292356463376]}

## Binary search (step 11) starts
Candidate diff: 0.0000488


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000488, mid=0.0000488, abs_max=0.10251016169786453
rel_dist={0: [-0.08991230398547434, 0.08991230398547431]}

## Binary search (step 12) starts
Candidate diff: 0.0000244


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000244, mid=0.0000244, abs_max=0.10251016169786453
rel_dist={0: [-0.08991199419703344, 0.08991199419702026]}

## Binary search (step 13) starts
Candidate diff: 0.0000122


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000122, mid=0.0000122, abs_max=0.10251016169786453
rel_dist={0: [-0.08991183930756332, 0.08991183930756333]}

## Binary search (step 14) starts
Candidate diff: 0.0000061


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000061, mid=0.0000061, abs_max=0.10251016169786453
rel_dist={0: [-0.08991176188726252, 0.08991176188726249]}

## Binary search (step 15) starts
Candidate diff: 0.0000031


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000031, mid=0.0000031, abs_max=0.10251016169786453
rel_dist={0: [-0.08991172476862014, 0.08991172321457086]}

## Binary search (step 16) starts
Candidate diff: 0.0000015


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000015, mid=0.0000015, abs_max=0.10251016169786453
rel_dist={0: [-0.08991170448511418, 0.08991173620782567]}

## Binary search (step 17) starts
Candidate diff: 0.0000008


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0000000, high=0.0000008, mid=0.0000008, abs_max=0.10251016169786453
rel_dist={0: [-0.08991169739850569, 0.08991171335425518]}

## Binary Search Result
Binary search time: 46.55 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1151.02 seconds

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0897690, upper bound: 0.0898732
time: 0.30 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0897690, upper bound: 0.0899608
time: 0.34 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.79 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -0.0897690, upper bound: 0.0898732
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -0.0897690, upper bound: 0.0899608

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0355980, 0.0491149, -0.0377331, 0.0562761, -0.0918742, 0.0868480
1: -0.0454062, 0.1038414, -0.0535909, 0.1269654, -0.1723715, 0.1574322
2: -0.1046092, 0.1549752, -0.1092526, 0.1708942, -0.2755034, 0.2642277
3: -0.0520488, 0.1242788, -0.0608371, 0.1547927, -0.2068415, 0.1851159
4: -0.1149560, 0.1775723, -0.1276938, 0.2019258, -0.3168817, 0.3052660

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896813, upper bound: 0.0896813
time: 0.30 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896813, upper bound: 0.0898732
time: 0.33 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0383316, 0.0603406, -0.0390887, 0.0634215, -0.1017531, 0.0994293
1: -0.0543237, 0.1360877, -0.0557757, 0.1414140, -0.1957377, 0.1918634
2: -0.1110429, 0.1767848, -0.1134923, 0.1820314, -0.2930743, 0.2902771
3: -0.0620208, 0.1679080, -0.0636387, 0.1747649, -0.2367857, 0.2315467
4: -0.1327446, 0.2087907, -0.1363116, 0.2151742, -0.3479187, 0.3451021

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898732, upper bound: 0.0897690
time: 0.33 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898732, upper bound: 0.0899608
time: 0.31 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.41 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -0.0896813, upper bound: 0.0896813
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -0.0896813, upper bound: 0.0898732
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -0.0898732, upper bound: 0.0897690
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 0, lower bound: -0.0898732, upper bound: 0.0899608

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0355980, 0.0491149, -0.0355980, 0.0491149, -0.0847130, 0.0847130
1: -0.0454062, 0.1038414, -0.0454062, 0.1038414, -0.1492476, 0.1492476
2: -0.1046092, 0.1549752, -0.1046092, 0.1549752, -0.2595844, 0.2595844
3: -0.0520488, 0.1242788, -0.0520488, 0.1242788, -0.1763276, 0.1763276
4: -0.1149560, 0.1775723, -0.1149560, 0.1775723, -0.2925282, 0.2925282

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896273, upper bound: 0.0896811
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896813, upper bound: 0.0896813
time: 0.30 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0355980, 0.0491149, -0.0380625, 0.0589265, -0.0945245, 0.0871774
1: -0.0454062, 0.1038414, -0.0539494, 0.1331241, -0.1785302, 0.1577908
2: -0.1046092, 0.1549752, -0.1101848, 0.1746560, -0.2792652, 0.2651600
3: -0.0520488, 0.1242788, -0.0615528, 0.1637270, -0.2157759, 0.1858316
4: -0.1149560, 0.1775723, -0.1309924, 0.2063214, -0.3212774, 0.3085647

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896273, upper bound: 0.0896811
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896273, upper bound: 0.0898717
time: 0.31 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0383316, 0.0603406, -0.0355980, 0.0491149, -0.0874465, 0.0959386
1: -0.0543237, 0.1360877, -0.0454062, 0.1038414, -0.1581650, 0.1814938
2: -0.1110429, 0.1767848, -0.1046092, 0.1549752, -0.2660180, 0.2813941
3: -0.0620208, 0.1679080, -0.0520488, 0.1242788, -0.1862996, 0.2199568
4: -0.1327446, 0.2087907, -0.1149560, 0.1775723, -0.3103168, 0.3237466

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898715, upper bound: 0.0897038
time: 0.33 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898717, upper bound: 0.0897578
time: 0.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0383316, 0.0603406, -0.0383316, 0.0603406, -0.0986722, 0.0986722
1: -0.0543237, 0.1360877, -0.0543237, 0.1360877, -0.1904113, 0.1904114
2: -0.1110429, 0.1767848, -0.1110429, 0.1767848, -0.2878277, 0.2878277
3: -0.0620208, 0.1679080, -0.0620208, 0.1679080, -0.2299288, 0.2299288
4: -0.1327446, 0.2087907, -0.1327446, 0.2087907, -0.3415352, 0.3415352

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898524, upper bound: 0.0898844
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898717, upper bound: 0.0899088
time: 0.32 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.19 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -0.0896273, upper bound: 0.0896811
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -0.0896813, upper bound: 0.0896813
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -0.0896273, upper bound: 0.0896811
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -0.0896273, upper bound: 0.0898717
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -0.0898715, upper bound: 0.0897038
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -0.0898717, upper bound: 0.0897578
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -0.0898524, upper bound: 0.0898844
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.19
Output dim: 0, lower bound: -0.0898717, upper bound: 0.0899088

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0355980, 0.0491149, -0.0780348, 0.0672179
1: -0.0364605, 0.0725038, -0.0454062, 0.1038414, -0.1403019, 0.1179100
2: -0.0836895, 0.1224094, -0.1046092, 0.1549752, -0.2386647, 0.2270187
3: -0.0432477, 0.0836505, -0.0520488, 0.1242788, -0.1675265, 0.1356993
4: -0.0884080, 0.1397506, -0.1149560, 0.1775723, -0.2659802, 0.2547066

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896271, upper bound: 0.0896271
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896271, upper bound: 0.0896811
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0341963, 0.0451829, -0.0355980, 0.0491149, -0.0833112, 0.0807809
1: -0.0427969, 0.0966445, -0.0454062, 0.1038414, -0.1466383, 0.1420507
2: -0.1001576, 0.1478489, -0.1046092, 0.1549752, -0.2551327, 0.2524582
3: -0.0493405, 0.1152938, -0.0520488, 0.1242788, -0.1736193, 0.1673426
4: -0.1093097, 0.1684246, -0.1149560, 0.1775723, -0.2868820, 0.2833806

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888442, upper bound: 0.0890278
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886358, upper bound: 0.0886358
time: 0.31 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0380625, 0.0589265, -0.0878463, 0.0696824
1: -0.0364605, 0.0725038, -0.0539494, 0.1331241, -0.1695846, 0.1264532
2: -0.0836895, 0.1224094, -0.1101848, 0.1746560, -0.2583454, 0.2325943
3: -0.0432477, 0.0836505, -0.0615528, 0.1637270, -0.2069747, 0.1452033
4: -0.0884080, 0.1397506, -0.1309924, 0.2063214, -0.2947294, 0.2707430

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896826, upper bound: 0.0898522
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896826, upper bound: 0.0898715
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0341963, 0.0451829, -0.0380625, 0.0589265, -0.0931227, 0.0832454
1: -0.0427969, 0.0966445, -0.0539494, 0.1331241, -0.1759210, 0.1505939
2: -0.1001576, 0.1478489, -0.1101848, 0.1746560, -0.2748135, 0.2580337
3: -0.0493405, 0.1152938, -0.0615528, 0.1637270, -0.2130676, 0.1768466
4: -0.1093097, 0.1684246, -0.1309924, 0.2063214, -0.3156311, 0.2994169

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0897366, upper bound: 0.0898524
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0897366, upper bound: 0.0898717
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0383316, 0.0603406, -0.0289199, 0.0316199, -0.0699515, 0.0892605
1: -0.0543237, 0.1360877, -0.0364605, 0.0725038, -0.1268275, 0.1725482
2: -0.1110429, 0.1767848, -0.0836895, 0.1224094, -0.2334523, 0.2604744
3: -0.0620208, 0.1679080, -0.0432477, 0.0836505, -0.1456713, 0.2111557
4: -0.1327446, 0.2087907, -0.0884080, 0.1397506, -0.2724952, 0.2971986

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898522, upper bound: 0.0896826
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898522, upper bound: 0.0897038
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0383316, 0.0603406, -0.0341963, 0.0451829, -0.0835145, 0.0945368
1: -0.0543237, 0.1360877, -0.0427969, 0.0966445, -0.1509682, 0.1788846
2: -0.1110429, 0.1767848, -0.1001576, 0.1478489, -0.2588917, 0.2769423
3: -0.0620208, 0.1679080, -0.0493405, 0.1152938, -0.1773146, 0.2172485
4: -0.1327446, 0.2087907, -0.1093097, 0.1684246, -0.3011691, 0.3181004

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898524, upper bound: 0.0897366
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898524, upper bound: 0.0897578
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0303880, 0.0381159, -0.0383316, 0.0603406, -0.0907286, 0.0764475
1: -0.0436567, 0.0986741, -0.0543237, 0.1360877, -0.1797444, 0.1529978
2: -0.0858032, 0.1358688, -0.1110429, 0.1767848, -0.2625880, 0.2469117
3: -0.0513058, 0.1197162, -0.0620208, 0.1679080, -0.2192139, 0.1817370
4: -0.1010720, 0.1610686, -0.1327446, 0.2087907, -0.3098626, 0.2938131

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894891, upper bound: 0.0882623
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885870, upper bound: 0.0881785
time: 0.31 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0352700, 0.0520305, -0.0383316, 0.0603406, -0.0956105, 0.0903621
1: -0.0497746, 0.1236877, -0.0543237, 0.1360877, -0.1858622, 0.1780114
2: -0.1017607, 0.1623120, -0.1110429, 0.1767848, -0.2785456, 0.2733549
3: -0.0576897, 0.1530293, -0.0620208, 0.1679080, -0.2255976, 0.2150501
4: -0.1224101, 0.1915598, -0.1327446, 0.2087907, -0.3312007, 0.3243043

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0897056, upper bound: 0.0888748
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
time: 0.33 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.61 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0896271, upper bound: 0.0896271
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0896271, upper bound: 0.0896811
IS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0888442, upper bound: 0.0890278
IS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0886358, upper bound: 0.0886358
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0896826, upper bound: 0.0898522
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0896826, upper bound: 0.0898715
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0897366, upper bound: 0.0898524
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0897366, upper bound: 0.0898717
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0898522, upper bound: 0.0896826
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0898522, upper bound: 0.0897038
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0898524, upper bound: 0.0897366
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0898524, upper bound: 0.0897578
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0894891, upper bound: 0.0882623
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0885870, upper bound: 0.0881785
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0897056, upper bound: 0.0888748
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0289199, 0.0316199, -0.0605398, 0.0605398
1: -0.0364605, 0.0725038, -0.0364605, 0.0725038, -0.1089643, 0.1089643
2: -0.0836895, 0.1224094, -0.0836895, 0.1224094, -0.2060990, 0.2060990
3: -0.0432477, 0.0836505, -0.0432477, 0.0836505, -0.1268981, 0.1268981
4: -0.0884080, 0.1397506, -0.0884080, 0.1397506, -0.2281586, 0.2281586

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894284, upper bound: 0.0895061
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889456, upper bound: 0.0888046
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885535, upper bound: 0.0885538
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0341963, 0.0451829, -0.0741027, 0.0658161
1: -0.0364605, 0.0725038, -0.0427969, 0.0966445, -0.1331051, 0.1153007
2: -0.0836895, 0.1224094, -0.1001576, 0.1478489, -0.2315385, 0.2225670
3: -0.0432477, 0.0836505, -0.0493405, 0.1152938, -0.1585415, 0.1329910
4: -0.0884080, 0.1397506, -0.1093097, 0.1684246, -0.2568326, 0.2490603

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889456, upper bound: 0.0888446
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885535, upper bound: 0.0886361
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0355980, 0.0491149, -0.0728816, 0.0562707
1: -0.0301767, 0.0481623, -0.0454062, 0.1038414, -0.1340180, 0.0935685
2: -0.0622640, 0.0955307, -0.1046092, 0.1549752, -0.2172392, 0.2001399
3: -0.0332325, 0.0495364, -0.0520488, 0.1242788, -0.1575113, 0.1015852
4: -0.0597730, 0.1052151, -0.1149560, 0.1775723, -0.2373453, 0.2201710

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888442, upper bound: 0.0889456
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888442, upper bound: 0.0889642
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0322836, 0.0334207, -0.0351903, 0.0471169, -0.0794005, 0.0686110
1: -0.0396060, 0.0710703, -0.0446699, 0.0999654, -0.1395714, 0.1157402
2: -0.0933639, 0.1287424, -0.1032541, 0.1513419, -0.2447057, 0.2319965
3: -0.0451630, 0.0779622, -0.0512630, 0.1188850, -0.1640480, 0.1292252
4: -0.0942796, 0.1455511, -0.1124166, 0.1733905, -0.2676701, 0.2579677

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884127, upper bound: 0.0886358
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884127, upper bound: 0.0886358
time: 0.32 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0301491, 0.0369835, -0.0659034, 0.0617690
1: -0.0364605, 0.0725038, -0.0433330, 0.0959707, -0.1324312, 0.1158367
2: -0.0836895, 0.1224094, -0.0850189, 0.1340068, -0.2176964, 0.2074283
3: -0.0432477, 0.0836505, -0.0508758, 0.1157976, -0.1590452, 0.1345263
4: -0.0884080, 0.1397506, -0.0994624, 0.1588748, -0.2472828, 0.2392130

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894839, upper bound: 0.0897163
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894730, upper bound: 0.0897311
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0350177, 0.0506750, -0.0795948, 0.0666376
1: -0.0364605, 0.0725038, -0.0494419, 0.1207996, -0.1572601, 0.1219457
2: -0.0836895, 0.1224094, -0.1009219, 0.1603026, -0.2439921, 0.2233313
3: -0.0432477, 0.0836505, -0.0572466, 0.1489364, -0.1921840, 0.1408971
4: -0.0884080, 0.1397506, -0.1207179, 0.1892169, -0.2776249, 0.2604685

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894839, upper bound: 0.0897506
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894730, upper bound: 0.0897497
time: 0.31 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0341963, 0.0451829, -0.0301491, 0.0369835, -0.0711798, 0.0753319
1: -0.0427969, 0.0966445, -0.0433330, 0.0959707, -0.1387676, 0.1399775
2: -0.1001576, 0.1478489, -0.0850189, 0.1340068, -0.2341644, 0.2328678
3: -0.0493405, 0.1152938, -0.0508758, 0.1157976, -0.1651381, 0.1661696
4: -0.1093097, 0.1684246, -0.0994624, 0.1588748, -0.2681845, 0.2678869

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890695, upper bound: 0.0892715
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891090, upper bound: 0.0890427
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0341963, 0.0451829, -0.0350177, 0.0506750, -0.0848712, 0.0802006
1: -0.0427969, 0.0966445, -0.0494419, 0.1207996, -0.1635965, 0.1460865
2: -0.1001576, 0.1478489, -0.1009219, 0.1603026, -0.2604602, 0.2487708
3: -0.0493405, 0.1152938, -0.0572466, 0.1489364, -0.1982769, 0.1725405
4: -0.1093097, 0.1684246, -0.1207179, 0.1892169, -0.2985266, 0.2891424

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890695, upper bound: 0.0892734
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891090, upper bound: 0.0890427
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0303880, 0.0381159, -0.0289199, 0.0316199, -0.0620079, 0.0670358
1: -0.0436567, 0.0986741, -0.0364605, 0.0725038, -0.1161605, 0.1351346
2: -0.0858032, 0.1358688, -0.0836895, 0.1224094, -0.2082126, 0.2195583
3: -0.0513058, 0.1197162, -0.0432477, 0.0836505, -0.1349563, 0.1629638
4: -0.1010720, 0.1610686, -0.0884080, 0.1397506, -0.2408226, 0.2494766

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0892715, upper bound: 0.0889486
time: 0.32 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890430, upper bound: 0.0889933
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0352700, 0.0520305, -0.0289199, 0.0316199, -0.0668899, 0.0809503
1: -0.0497746, 0.1236877, -0.0364605, 0.0725038, -0.1222784, 0.1601482
2: -0.1017607, 0.1623120, -0.0836895, 0.1224094, -0.2241702, 0.2460016
3: -0.0576897, 0.1530293, -0.0432477, 0.0836505, -0.1413402, 0.1962770
4: -0.1224101, 0.1915598, -0.0884080, 0.1397506, -0.2621607, 0.2799676

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0897163, upper bound: 0.0895051
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0897311, upper bound: 0.0894730
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0303880, 0.0381159, -0.0341963, 0.0451829, -0.0755709, 0.0723121
1: -0.0436567, 0.0986741, -0.0427969, 0.0966445, -0.1403012, 0.1414710
2: -0.0858032, 0.1358688, -0.1001576, 0.1478489, -0.2336521, 0.2360264
3: -0.0513058, 0.1197162, -0.0493405, 0.1152938, -0.1665996, 0.1690567
4: -0.1010720, 0.1610686, -0.1093097, 0.1684246, -0.2694965, 0.2703783

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0892715, upper bound: 0.0890695
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890427, upper bound: 0.0891090
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0352700, 0.0520305, -0.0341963, 0.0451829, -0.0804528, 0.0862267
1: -0.0497746, 0.1236877, -0.0427969, 0.0966445, -0.1464191, 0.1664846
2: -0.1017607, 0.1623120, -0.1001576, 0.1478489, -0.2496097, 0.2624695
3: -0.0576897, 0.1530293, -0.0493405, 0.1152938, -0.1729835, 0.2023698
4: -0.1224101, 0.1915598, -0.1093097, 0.1684246, -0.2908346, 0.3008695

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0892715, upper bound: 0.0889618
time: 0.33 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890427, upper bound: 0.0890129
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0303880, 0.0381159, -0.0358490, 0.0487337, -0.0791218, 0.0739649
1: -0.0436567, 0.0986741, -0.0507856, 0.1154432, -0.1590999, 0.1494596
2: -0.0858032, 0.1358688, -0.1024819, 0.1582004, -0.2440036, 0.2383507
3: -0.0513058, 0.1197162, -0.0580275, 0.1404007, -0.1917065, 0.1777436
4: -0.1010720, 0.1610686, -0.1195223, 0.1871233, -0.2881952, 0.2805909

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885870, upper bound: 0.0881785
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885870, upper bound: 0.0881785
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0279236, 0.0336640, -0.0262249, 0.0476892, -0.0756128, 0.0598889
1: -0.0407126, 0.0920340, -0.0560731, 0.1461119, -0.1868245, 0.1481072
2: -0.0783510, 0.1263166, -0.0741902, 0.1451171, -0.2234681, 0.2005068
3: -0.0486229, 0.1124485, -0.0633892, 0.1894719, -0.2380949, 0.1758377
4: -0.0940725, 0.1500419, -0.1138687, 0.1800817, -0.2741542, 0.2639106

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885870, upper bound: 0.0881785
time: 0.32 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885870, upper bound: 0.0881785
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0352700, 0.0520305, -0.0358490, 0.0487337, -0.0840037, 0.0878795
1: -0.0497746, 0.1236877, -0.0507856, 0.1154432, -0.1652177, 0.1744732
2: -0.1017607, 0.1623120, -0.1024819, 0.1582004, -0.2599612, 0.2647940
3: -0.0576897, 0.1530293, -0.0580275, 0.1404007, -0.1980903, 0.2110567
4: -0.1224101, 0.1915598, -0.1195223, 0.1871233, -0.3095332, 0.3110821

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0330304, 0.0473732, -0.0262249, 0.0476892, -0.0807196, 0.0735981
1: -0.0469812, 0.1166384, -0.0560731, 0.1461119, -0.1930932, 0.1727115
2: -0.0949624, 0.1530344, -0.0741902, 0.1451171, -0.2400795, 0.2272245
3: -0.0551836, 0.1444830, -0.0633892, 0.1894719, -0.2446555, 0.2078722
4: -0.1155490, 0.1808904, -0.1138687, 0.1800817, -0.2956307, 0.2947589

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
time: 0.34 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.67 seconds
IS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0889456, upper bound: 0.0888046
IS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0885535, upper bound: 0.0885538
IS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0889456, upper bound: 0.0888446
IS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0885535, upper bound: 0.0886361
IS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0888442, upper bound: 0.0889456
IS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0888442, upper bound: 0.0889642
IS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0884127, upper bound: 0.0886358
IS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0884127, upper bound: 0.0886358
IS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0894839, upper bound: 0.0897163
IS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0894730, upper bound: 0.0897311
IS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0894839, upper bound: 0.0897506
IS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0894730, upper bound: 0.0897497
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0890695, upper bound: 0.0892715
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0891090, upper bound: 0.0890427
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0890695, upper bound: 0.0892734
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0891090, upper bound: 0.0890427
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0892715, upper bound: 0.0889486
IS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0890430, upper bound: 0.0889933
IS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0897163, upper bound: 0.0895051
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0897311, upper bound: 0.0894730
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0892715, upper bound: 0.0890695
IS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0890427, upper bound: 0.0891090
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0892715, upper bound: 0.0889618
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0890427, upper bound: 0.0890129
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0885870, upper bound: 0.0881785
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0885870, upper bound: 0.0881785
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0885870, upper bound: 0.0881785
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0885870, upper bound: 0.0881785
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.67
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871

## BFS IS instance: IS_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0189038, 0.0158268, -0.0447467, 0.0505237
1: -0.0364605, 0.0725038, -0.0248756, 0.0358514, -0.0723119, 0.0973794
2: -0.0836895, 0.1224094, -0.0468024, 0.0796161, -0.1633056, 0.1692118
3: -0.0432477, 0.0836505, -0.0280160, 0.0339445, -0.0771922, 0.1116665
4: -0.0884080, 0.1397506, -0.0429375, 0.0872777, -0.1756857, 0.1826881

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886988, upper bound: 0.0887597
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885925, upper bound: 0.0885696
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0284522, 0.0300982, -0.0274795, 0.0253483, -0.0538005, 0.0575777
1: -0.0356955, 0.0692185, -0.0339634, 0.0554254, -0.0911208, 0.1031820
2: -0.0822833, 0.1195021, -0.0789161, 0.1109374, -0.1932207, 0.1984182
3: -0.0424577, 0.0790637, -0.0402230, 0.0574996, -0.0999574, 0.1192867
4: -0.0860603, 0.1362391, -0.0770320, 0.1258029, -0.2118632, 0.2132711

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883070, upper bound: 0.0885059
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882591, upper bound: 0.0882591
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0237667, 0.0206726, -0.0495925, 0.0553866
1: -0.0364605, 0.0725038, -0.0301767, 0.0481623, -0.0846229, 0.1026805
2: -0.0836895, 0.1224094, -0.0622640, 0.0955307, -0.1792202, 0.1846735
3: -0.0432477, 0.0836505, -0.0332325, 0.0495364, -0.0927840, 0.1168830
4: -0.0884080, 0.1397506, -0.0597730, 0.1052151, -0.1936230, 0.1995236

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886988, upper bound: 0.0887986
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884127, upper bound: 0.0884950
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884127, upper bound: 0.0886361
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0284522, 0.0300982, -0.0322836, 0.0334207, -0.0618729, 0.0623818
1: -0.0356955, 0.0692185, -0.0396060, 0.0710703, -0.1067657, 0.1088246
2: -0.0822833, 0.1195021, -0.0933639, 0.1287424, -0.2110257, 0.2128660
3: -0.0424577, 0.0790637, -0.0451630, 0.0779622, -0.1204199, 0.1242266
4: -0.0860603, 0.1362391, -0.0942796, 0.1455511, -0.2316114, 0.2305187

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883067, upper bound: 0.0885881
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885535, upper bound: 0.0884950
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885535, upper bound: 0.0886361
time: 0.30 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0289199, 0.0316199, -0.0553866, 0.0495925
1: -0.0301767, 0.0481623, -0.0364605, 0.0725038, -0.1026805, 0.0846229
2: -0.0622640, 0.0955307, -0.0836895, 0.1224094, -0.1846735, 0.1792202
3: -0.0332325, 0.0495364, -0.0432477, 0.0836505, -0.1168830, 0.0927840
4: -0.0597730, 0.1052151, -0.0884080, 0.1397506, -0.1995236, 0.1936230

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887054, upper bound: 0.0889456
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887054, upper bound: 0.0889456
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0341963, 0.0451829, -0.0689495, 0.0548689
1: -0.0301767, 0.0481623, -0.0427969, 0.0966445, -0.1268212, 0.0909593
2: -0.0622640, 0.0955307, -0.1001576, 0.1478489, -0.2101130, 0.1956883
3: -0.0332325, 0.0495364, -0.0493405, 0.1152938, -0.1485263, 0.0988769
4: -0.0597730, 0.1052151, -0.1093097, 0.1684246, -0.2281975, 0.2145248

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887054, upper bound: 0.0889642
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886665, upper bound: 0.0889642
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0322836, 0.0334207, -0.0246740, 0.0221408, -0.0544243, 0.0580946
1: -0.0396060, 0.0710703, -0.0320872, 0.0524673, -0.0920733, 0.1031575
2: -0.0933639, 0.1287424, -0.0651265, 0.1000648, -0.1934287, 0.1938689
3: -0.0451630, 0.0779622, -0.0352156, 0.0550946, -0.1002576, 0.1131778
4: -0.0942796, 0.1455511, -0.0637753, 0.1115354, -0.2058151, 0.2093264

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0885535
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0885564
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0322836, 0.0334207, -0.0342600, 0.0395509, -0.0718345, 0.0676807
1: -0.0396060, 0.0710703, -0.0425888, 0.0814996, -0.1211056, 0.1136591
2: -0.0933639, 0.1287424, -0.0995706, 0.1393745, -0.2327384, 0.2283130
3: -0.0451630, 0.0779622, -0.0484002, 0.0913292, -0.1364921, 0.1263624
4: -0.0942796, 0.1455511, -0.1022166, 0.1587072, -0.2529868, 0.2477677

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0885535
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0885564
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0242503, 0.0299525, -0.0588724, 0.0558702
1: -0.0364605, 0.0725038, -0.0366645, 0.0839245, -0.1203851, 0.1091683
2: -0.0836895, 0.1224094, -0.0667591, 0.1147048, -0.1983943, 0.1891686
3: -0.0432477, 0.0836505, -0.0451630, 0.1024292, -0.1456769, 0.1288135
4: -0.0884080, 0.1397506, -0.0822306, 0.1366947, -0.2251027, 0.2219812

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894806, upper bound: 0.0897163
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894806, upper bound: 0.0897163
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0291563, 0.0336461, -0.0625660, 0.0607762
1: -0.0364605, 0.0725038, -0.0408317, 0.0895138, -0.1259744, 0.1133355
2: -0.0836895, 0.1224094, -0.0816707, 0.1277841, -0.2114736, 0.2040801
3: -0.0432477, 0.0836505, -0.0488912, 0.1076730, -0.1509207, 0.1325417
4: -0.0884080, 0.1397506, -0.0948198, 0.1511557, -0.2395636, 0.2345704

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886940, upper bound: 0.0891682
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887542, upper bound: 0.0889951
time: 0.31 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0295995, 0.0398316, -0.0687515, 0.0612194
1: -0.0364605, 0.0725038, -0.0433962, 0.1045852, -0.1410457, 0.1159000
2: -0.0836895, 0.1224094, -0.0837515, 0.1380744, -0.2217639, 0.2061609
3: -0.0432477, 0.0836505, -0.0519528, 0.1288300, -0.1720776, 0.1356032
4: -0.0884080, 0.1397506, -0.1026442, 0.1638758, -0.2522838, 0.2423948

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894730, upper bound: 0.0897497
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894730, upper bound: 0.0897497
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0337088, 0.0468724, -0.0757922, 0.0653287
1: -0.0364605, 0.0725038, -0.0466997, 0.1138791, -0.1503397, 0.1192035
2: -0.0836895, 0.1224094, -0.0970538, 0.1534397, -0.2371292, 0.2194632
3: -0.0432477, 0.0836505, -0.0549959, 0.1400864, -0.1833341, 0.1386464
4: -0.0884080, 0.1397506, -0.1155892, 0.1808250, -0.2692329, 0.2553398

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885969, upper bound: 0.0891538
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887073, upper bound: 0.0890131
time: 0.32 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0301491, 0.0369835, -0.0607502, 0.0508217
1: -0.0301767, 0.0481623, -0.0433330, 0.0959707, -0.1261474, 0.0914953
2: -0.0622640, 0.0955307, -0.0850189, 0.1340068, -0.1962709, 0.1805496
3: -0.0332325, 0.0495364, -0.0508758, 0.1157976, -0.1490301, 0.1004122
4: -0.0597730, 0.1052151, -0.0994624, 0.1588748, -0.2186479, 0.2046774

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890720, upper bound: 0.0892677
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886306, upper bound: 0.0888775
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886306, upper bound: 0.0889019
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0322836, 0.0334207, -0.0295164, 0.0350648, -0.0673484, 0.0629370
1: -0.0396060, 0.0710703, -0.0423116, 0.0919634, -0.1315695, 0.1133818
2: -0.0933639, 0.1287424, -0.0829426, 0.1303789, -0.2237428, 0.2116850
3: -0.0451630, 0.0779622, -0.0497846, 0.1105294, -0.1556924, 0.1277468
4: -0.0942796, 0.1455511, -0.0965452, 0.1544582, -0.2487379, 0.2420962

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891110, upper bound: 0.0890427
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886306, upper bound: 0.0890183
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886306, upper bound: 0.0890427
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0350177, 0.0506750, -0.0744417, 0.0556903
1: -0.0301767, 0.0481623, -0.0494419, 0.1207996, -0.1509763, 0.0976043
2: -0.0622640, 0.0955307, -0.1009219, 0.1603026, -0.2225667, 0.1964526
3: -0.0332325, 0.0495364, -0.0572466, 0.1489364, -0.1821689, 0.1067830
4: -0.0597730, 0.1052151, -0.1207179, 0.1892169, -0.2489900, 0.2259330

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890090, upper bound: 0.0892537
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886439, upper bound: 0.0888775
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886439, upper bound: 0.0889019
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0322836, 0.0334207, -0.0343857, 0.0484573, -0.0807409, 0.0678063
1: -0.0396060, 0.0710703, -0.0484488, 0.1164093, -0.1560153, 0.1195190
2: -0.0933639, 0.1287424, -0.0989168, 0.1564851, -0.2498490, 0.2276592
3: -0.0451630, 0.0779622, -0.0562381, 0.1430612, -0.1882242, 0.1342002
4: -0.0942796, 0.1455511, -0.1177716, 0.1846826, -0.2789622, 0.2633227

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890690, upper bound: 0.0890427
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886439, upper bound: 0.0890183
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886439, upper bound: 0.0890427
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0303880, 0.0381159, -0.0189038, 0.0158268, -0.0462148, 0.0570197
1: -0.0436567, 0.0986741, -0.0248756, 0.0358514, -0.0795081, 0.1235497
2: -0.0858032, 0.1358688, -0.0468024, 0.0796161, -0.1654193, 0.1826712
3: -0.0513058, 0.1197162, -0.0280160, 0.0339445, -0.0852503, 0.1477322
4: -0.1010720, 0.1610686, -0.0429375, 0.0872777, -0.1883497, 0.2040060

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891720, upper bound: 0.0886952
time: 0.34 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891682, upper bound: 0.0886940
time: 0.37 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0297309, 0.0360747, -0.0274795, 0.0253483, -0.0550791, 0.0635542
1: -0.0426078, 0.0943700, -0.0339634, 0.0554254, -0.0980332, 0.1283334
2: -0.0836866, 0.1320749, -0.0789161, 0.1109374, -0.1946240, 0.2109910
3: -0.0501978, 0.1139639, -0.0402230, 0.0574996, -0.1076974, 0.1541869
4: -0.0979943, 0.1564795, -0.0770320, 0.1258029, -0.2237972, 0.2335116

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889950, upper bound: 0.0887542
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889951, upper bound: 0.0887542
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0298293, 0.0411537, -0.0289199, 0.0316199, -0.0614492, 0.0700736
1: -0.0437109, 0.1075221, -0.0364605, 0.0725038, -0.1162146, 0.1439826
2: -0.0845616, 0.1400728, -0.0836895, 0.1224094, -0.2069710, 0.2237623
3: -0.0523840, 0.1329535, -0.0432477, 0.0836505, -0.1360344, 0.1762011
4: -0.1043052, 0.1662240, -0.0884080, 0.1397506, -0.2440558, 0.2546320

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891797, upper bound: 0.0886866
time: 0.36 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890131, upper bound: 0.0887520
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0339535, 0.0482027, -0.0289199, 0.0316199, -0.0655734, 0.0771226
1: -0.0470253, 0.1167137, -0.0364605, 0.0725038, -0.1195291, 0.1531743
2: -0.0978653, 0.1553957, -0.0836895, 0.1224094, -0.2202748, 0.2390852
3: -0.0554273, 0.1441097, -0.0432477, 0.0836505, -0.1390777, 0.1873574
4: -0.1172390, 0.1831006, -0.0884080, 0.1397506, -0.2569896, 0.2715085

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891537, upper bound: 0.0885969
time: 0.34 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890131, upper bound: 0.0887073
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0303880, 0.0381159, -0.0237667, 0.0206726, -0.0510606, 0.0618826
1: -0.0436567, 0.0986741, -0.0301767, 0.0481623, -0.0918190, 0.1288508
2: -0.0858032, 0.1358688, -0.0622640, 0.0955307, -0.1813339, 0.1981328
3: -0.0513058, 0.1197162, -0.0332325, 0.0495364, -0.1008422, 0.1529487
4: -0.1010720, 0.1610686, -0.0597730, 0.1052151, -0.2062870, 0.2208416

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0892677, upper bound: 0.0890720
time: 0.36 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888775, upper bound: 0.0886306
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888775, upper bound: 0.0890720
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0297309, 0.0360747, -0.0322836, 0.0334207, -0.0631515, 0.0683582
1: -0.0426078, 0.0943700, -0.0396060, 0.0710703, -0.1136781, 0.1339760
2: -0.0836866, 0.1320749, -0.0933639, 0.1287424, -0.2124290, 0.2254388
3: -0.0501978, 0.1139639, -0.0451630, 0.0779622, -0.1281600, 0.1591269
4: -0.0979943, 0.1564795, -0.0942796, 0.1455511, -0.2435454, 0.2507591

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890427, upper bound: 0.0891110
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890183, upper bound: 0.0886306
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890183, upper bound: 0.0886306
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0352700, 0.0520305, -0.0237667, 0.0206726, -0.0559426, 0.0757972
1: -0.0497746, 0.1236877, -0.0301767, 0.0481623, -0.0979369, 0.1538644
2: -0.1017607, 0.1623120, -0.0622640, 0.0955307, -0.1972914, 0.2245761
3: -0.0576897, 0.1530293, -0.0332325, 0.0495364, -0.1072261, 0.1862619
4: -0.1224101, 0.1915598, -0.0597730, 0.1052151, -0.2276252, 0.2513328

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0892537, upper bound: 0.0888673
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888892, upper bound: 0.0885952
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888892, upper bound: 0.0885952
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0346291, 0.0497373, -0.0322836, 0.0334207, -0.0680498, 0.0820208
1: -0.0487626, 0.1190682, -0.0396060, 0.0710703, -0.1198329, 0.1586742
2: -0.0997330, 0.1583815, -0.0933639, 0.1287424, -0.2284754, 0.2517453
3: -0.0566554, 0.1468186, -0.0451630, 0.0779622, -0.1346176, 0.1919816
4: -0.1193668, 0.1868879, -0.0942796, 0.1455511, -0.2649179, 0.2811675

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890607, upper bound: 0.0889498
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890300, upper bound: 0.0885952
time: 0.34 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890300, upper bound: 0.0890129
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0281139, 0.0310250, -0.0358490, 0.0487337, -0.0768476, 0.0668740
1: -0.0406024, 0.0818455, -0.0507856, 0.1154432, -0.1560455, 0.1326310
2: -0.0773617, 0.1205507, -0.1024819, 0.1582004, -0.2355621, 0.2230326
3: -0.0474901, 0.0966178, -0.0580275, 0.1404007, -0.1878907, 0.1546452
4: -0.0884082, 0.1426494, -0.1195223, 0.1871233, -0.2755313, 0.2621717

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894555, upper bound: 0.0882342
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894555, upper bound: 0.0882623
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0200956, 0.0350137, -0.0358490, 0.0487337, -0.0688294, 0.0708628
1: -0.0474771, 0.1145338, -0.0507856, 0.1154432, -0.1629202, 0.1653194
2: -0.0544097, 0.1152232, -0.1024819, 0.1582004, -0.2126101, 0.2177052
3: -0.0536702, 0.1447884, -0.0580275, 0.1404007, -0.1940709, 0.2028158
4: -0.0832517, 0.1436815, -0.1195223, 0.1871233, -0.2703749, 0.2632038

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888010, upper bound: 0.0877759
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889125, upper bound: 0.0877858
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0281048, 0.0310157, -0.0262249, 0.0476892, -0.0757940, 0.0572406
1: -0.0405944, 0.0818225, -0.0560731, 0.1461119, -0.1867063, 0.1378957
2: -0.0773178, 0.1205083, -0.0741902, 0.1451171, -0.2224349, 0.1946985
3: -0.0474758, 0.0965890, -0.0633892, 0.1894719, -0.2369477, 0.1599782
4: -0.0883616, 0.1425957, -0.1138687, 0.1800817, -0.2684433, 0.2564643

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885755, upper bound: 0.0881717
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885870, upper bound: 0.0881785
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0200956, 0.0350137, -0.0262249, 0.0476892, -0.0677848, 0.0612386
1: -0.0474771, 0.1145338, -0.0560731, 0.1461119, -0.1935890, 0.1706070
2: -0.0544097, 0.1152232, -0.0741902, 0.1451171, -0.1995267, 0.1894134
3: -0.0536702, 0.1447884, -0.0633892, 0.1894719, -0.2431421, 0.2081776
4: -0.0832517, 0.1436815, -0.1138687, 0.1800817, -0.2633333, 0.2575501

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879745, upper bound: 0.0879238
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879745, upper bound: 0.0881785
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0326413, 0.0406024, -0.0358490, 0.0487337, -0.0813750, 0.0764514
1: -0.0464028, 0.1029233, -0.0507856, 0.1154432, -0.1618460, 0.1537088
2: -0.0931736, 0.1439410, -0.1024819, 0.1582004, -0.2513740, 0.2464229
3: -0.0537068, 0.1251042, -0.0580275, 0.1404007, -0.1941074, 0.1831317
4: -0.1092366, 0.1700594, -0.1195223, 0.1871233, -0.2963600, 0.2895818

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896720, upper bound: 0.0888467
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896720, upper bound: 0.0888467
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0252827, 0.0445866, -0.0358490, 0.0487337, -0.0740165, 0.0804356
1: -0.0539653, 0.1393512, -0.0507856, 0.1154432, -0.1694084, 0.1901368
2: -0.0712305, 0.1388832, -0.1024819, 0.1582004, -0.2294308, 0.2413651
3: -0.0606718, 0.1808298, -0.0580275, 0.1404007, -0.2010725, 0.2388572
4: -0.1092791, 0.1714957, -0.1195223, 0.1871233, -0.2964023, 0.2910181

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0897000, upper bound: 0.0888634
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0897044, upper bound: 0.0888748
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0326413, 0.0406024, -0.0262249, 0.0476892, -0.0803305, 0.0668272
1: -0.0464028, 0.1029233, -0.0560731, 0.1461119, -0.1925147, 0.1589964
2: -0.0931736, 0.1439410, -0.0741902, 0.1451171, -0.2382907, 0.2181312
3: -0.0537068, 0.1251042, -0.0633892, 0.1894719, -0.2431787, 0.1884934
4: -0.1092366, 0.1700594, -0.1138687, 0.1800817, -0.2893183, 0.2839281

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879593
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880213
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0252827, 0.0445866, -0.0262249, 0.0476892, -0.0729719, 0.0708114
1: -0.0539653, 0.1393512, -0.0560731, 0.1461119, -0.2000772, 0.1954244
2: -0.0712305, 0.1388832, -0.0741902, 0.1451171, -0.2163476, 0.2130734
3: -0.0606718, 0.1808298, -0.0633892, 0.1894719, -0.2501436, 0.2442190
4: -0.1092791, 0.1714957, -0.1138687, 0.1800817, -0.2893607, 0.2853644

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879382, upper bound: 0.0879721
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880213
time: 0.34 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.48 seconds
IS_A1_B1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0886988, upper bound: 0.0887597
IS_A1_B1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0885925, upper bound: 0.0885696
IS_A1_B1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0883070, upper bound: 0.0885059
IS_A1_B1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0882591, upper bound: 0.0882591
IS_A1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0884127, upper bound: 0.0884950
IS_A1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0884127, upper bound: 0.0886361
IS_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0885535, upper bound: 0.0884950
IS_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0885535, upper bound: 0.0886361
IS_A1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0887054, upper bound: 0.0889456
IS_A1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0887054, upper bound: 0.0889456
IS_A1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0887054, upper bound: 0.0889642
IS_A1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0886665, upper bound: 0.0889642
IS_A1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0885535
IS_A1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0885564
IS_A1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0885535
IS_A1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0885564
IS_A1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0894806, upper bound: 0.0897163
IS_A1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0894806, upper bound: 0.0897163
IS_A1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0886940, upper bound: 0.0891682
IS_A1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0887542, upper bound: 0.0889951
IS_A1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0894730, upper bound: 0.0897497
IS_A1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0894730, upper bound: 0.0897497
IS_A1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0885969, upper bound: 0.0891538
IS_A1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0887073, upper bound: 0.0890131
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0886306, upper bound: 0.0888775
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0886306, upper bound: 0.0889019
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0886306, upper bound: 0.0890183
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0886306, upper bound: 0.0890427
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0886439, upper bound: 0.0888775
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0886439, upper bound: 0.0889019
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0886439, upper bound: 0.0890183
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0886439, upper bound: 0.0890427
IS_A2_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0891720, upper bound: 0.0886952
IS_A2_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0891682, upper bound: 0.0886940
IS_A2_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0889950, upper bound: 0.0887542
IS_A2_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0889951, upper bound: 0.0887542
IS_A2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0891797, upper bound: 0.0886866
IS_A2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0890131, upper bound: 0.0887520
IS_A2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0891537, upper bound: 0.0885969
IS_A2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0890131, upper bound: 0.0887073
IS_A2_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0888775, upper bound: 0.0886306
IS_A2_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0888775, upper bound: 0.0890720
IS_A2_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0890183, upper bound: 0.0886306
IS_A2_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0890183, upper bound: 0.0886306
IS_A2_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0888892, upper bound: 0.0885952
IS_A2_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0888892, upper bound: 0.0885952
IS_A2_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0890300, upper bound: 0.0885952
IS_A2_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0890300, upper bound: 0.0890129
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0894555, upper bound: 0.0882342
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0894555, upper bound: 0.0882623
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0888010, upper bound: 0.0877759
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0889125, upper bound: 0.0877858
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0885755, upper bound: 0.0881717
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0885870, upper bound: 0.0881785
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0879745, upper bound: 0.0879238
IS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0879745, upper bound: 0.0881785
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0896720, upper bound: 0.0888467
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0896720, upper bound: 0.0888467
IS_A2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0897000, upper bound: 0.0888634
IS_A2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0897044, upper bound: 0.0888748
IS_A2_B2_A2_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879593
IS_A2_B2_A2_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880213
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0879382, upper bound: 0.0879721
IS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.48
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880213

## BFS IS instance: IS_A1_B1_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0183061, 0.0140802, -0.0430001, 0.0499260
1: -0.0364605, 0.0725038, -0.0240727, 0.0309017, -0.0673622, 0.0965765
2: -0.0836895, 0.1224094, -0.0450367, 0.0758257, -0.1595152, 0.1674461
3: -0.0432477, 0.0836505, -0.0263939, 0.0274213, -0.0706689, 0.1100443
4: -0.0884080, 0.1397506, -0.0390864, 0.0815583, -0.1699663, 0.1788370

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885925, upper bound: 0.0885696
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885925, upper bound: 0.0885696
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0182536, 0.0148930, -0.0438129, 0.0498735
1: -0.0364605, 0.0725038, -0.0242180, 0.0328669, -0.0693275, 0.0967218
2: -0.0836895, 0.1224094, -0.0444218, 0.0763963, -0.1600859, 0.1668313
3: -0.0432477, 0.0836505, -0.0269134, 0.0300028, -0.0732505, 0.1105639
4: -0.0884080, 0.1397506, -0.0399020, 0.0830934, -0.1715014, 0.1796526

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885925, upper bound: 0.0885696
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885925, upper bound: 0.0885696
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0284522, 0.0300982, -0.0235625, 0.0223077, -0.0507599, 0.0536608
1: -0.0356955, 0.0692185, -0.0298257, 0.0489782, -0.0846737, 0.0990443
2: -0.0822833, 0.1195021, -0.0665481, 0.0995724, -0.1818557, 0.1860502
3: -0.0424577, 0.0790637, -0.0366166, 0.0498813, -0.0923391, 0.1156802
4: -0.0860603, 0.1362391, -0.0645988, 0.1130393, -0.1990996, 0.2008379

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881659, upper bound: 0.0883486
time: 0.38 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881659, upper bound: 0.0885059
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0284522, 0.0300982, -0.0268614, 0.0244703, -0.0529226, 0.0569597
1: -0.0356955, 0.0692185, -0.0328907, 0.0524811, -0.0881766, 0.1021092
2: -0.0822833, 0.1195021, -0.0769134, 0.1082844, -0.1905677, 0.1964155
3: -0.0424577, 0.0790637, -0.0390857, 0.0535550, -0.0960127, 0.1181494
4: -0.0860603, 0.1362391, -0.0741939, 0.1224238, -0.2084841, 0.2104329

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882591, upper bound: 0.0882591
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882591, upper bound: 0.0882591
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0189038, 0.0158268, -0.0237667, 0.0206726, -0.0395764, 0.0395935
1: -0.0248756, 0.0358514, -0.0301767, 0.0481623, -0.0730379, 0.0660281
2: -0.0468024, 0.0796161, -0.0622640, 0.0955307, -0.1423331, 0.1418801
3: -0.0280160, 0.0339445, -0.0332325, 0.0495364, -0.0775524, 0.0671770
4: -0.0429375, 0.0872777, -0.0597730, 0.1052151, -0.1481525, 0.1470508

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886988, upper bound: 0.0886413
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 36

Time for candidate selection: 2.42 seconds

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 49
type: A, layer: 5, pos: 49
type: B, layer: 5, pos: 33
type: A, layer: 5, pos: 33
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 1
type: B, layer: 5, pos: 23
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 1
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 23
type: A, layer: 5, pos: 42
type: A, layer: 5, pos: 16
type: B, layer: 5, pos: 42
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 26
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 16
type: B, layer: 5, pos: 14

Time for candidate selection: 6.33 seconds

### Candidate
type: B, layer: 5, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0874344, upper bound: 0.0866909
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884095, upper bound: 0.0882805
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0274795, 0.0253483, -0.0237667, 0.0206726, -0.0481521, 0.0491150
1: -0.0339634, 0.0554254, -0.0301767, 0.0481623, -0.0821258, 0.0856021
2: -0.0789161, 0.1109374, -0.0622640, 0.0955307, -0.1744468, 0.1732014
3: -0.0402230, 0.0574996, -0.0332325, 0.0495364, -0.0897594, 0.0907322
4: -0.0770320, 0.1258029, -0.0597730, 0.1052151, -0.1822471, 0.1855760

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886988, upper bound: 0.0887986
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36

Time for candidate selection: 2.29 seconds

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 33
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 23
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 23
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 34
type: A, layer: 5, pos: 16
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 26
type: B, layer: 5, pos: 49
type: A, layer: 5, pos: 1
type: B, layer: 5, pos: 1
type: B, layer: 5, pos: 16
type: B, layer: 5, pos: 14
type: B, layer: 5, pos: 42

Time for candidate selection: 5.90 seconds

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0874344, upper bound: 0.0867183
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884095, upper bound: 0.0884208
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0189038, 0.0158268, -0.0322836, 0.0334207, -0.0523245, 0.0481104
1: -0.0248756, 0.0358514, -0.0396060, 0.0710703, -0.0959459, 0.0754574
2: -0.0468024, 0.0796161, -0.0933639, 0.1287424, -0.1755448, 0.1729800
3: -0.0280160, 0.0339445, -0.0451630, 0.0779622, -0.1059782, 0.0791075
4: -0.0429375, 0.0872777, -0.0942796, 0.1455511, -0.1884885, 0.1815574

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881659, upper bound: 0.0884308
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 21

Time for candidate selection: 2.17 seconds

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 33
type: A, layer: 5, pos: 33
type: A, layer: 5, pos: 26
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 26
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 1
type: B, layer: 5, pos: 23
type: B, layer: 5, pos: 34
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 34
type: A, layer: 5, pos: 16
type: A, layer: 5, pos: 23
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 49
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 28
type: A, layer: 5, pos: 42

Time for candidate selection: 5.44 seconds

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0868762, upper bound: 0.0863725
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878513, upper bound: 0.0879620
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0274795, 0.0253483, -0.0322836, 0.0334207, -0.0609002, 0.0576319
1: -0.0339634, 0.0554254, -0.0396060, 0.0710703, -0.1050337, 0.0950314
2: -0.0789161, 0.1109374, -0.0933639, 0.1287424, -0.2076585, 0.2043013
3: -0.0402230, 0.0574996, -0.0451630, 0.0779622, -0.1181852, 0.1026626
4: -0.0770320, 0.1258029, -0.0942796, 0.1455511, -0.2225831, 0.2200826

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881659, upper bound: 0.0885881
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 11

Time for candidate selection: 2.04 seconds

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 33
type: A, layer: 5, pos: 33
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 26
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 15
type: B, layer: 5, pos: 23
type: B, layer: 5, pos: 26
type: A, layer: 5, pos: 23
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 28
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 16
type: A, layer: 5, pos: 1
type: B, layer: 5, pos: 16
type: B, layer: 5, pos: 1

Time for candidate selection: 5.24 seconds

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0868762, upper bound: 0.0864058
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878513, upper bound: 0.0881033
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0189038, 0.0158268, -0.0395935, 0.0395764
1: -0.0301767, 0.0481623, -0.0248756, 0.0358514, -0.0660281, 0.0730379
2: -0.0622640, 0.0955307, -0.0468024, 0.0796161, -0.1418801, 0.1423331
3: -0.0332325, 0.0495364, -0.0280160, 0.0339445, -0.0671770, 0.0775524
4: -0.0597730, 0.1052151, -0.0429375, 0.0872777, -0.1470508, 0.1481525

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886413, upper bound: 0.0886988
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 36

Time for candidate selection: 2.15 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 49
type: B, layer: 5, pos: 49
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 33
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 23
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 1
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 23
type: B, layer: 5, pos: 42
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 42
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 26
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 16
type: A, layer: 5, pos: 14

Time for candidate selection: 5.50 seconds

### Candidate
type: A, layer: 5, pos: 49

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 49

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0866909, upper bound: 0.0874344
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882805, upper bound: 0.0884095
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0274795, 0.0253483, -0.0491150, 0.0481521
1: -0.0301767, 0.0481623, -0.0339634, 0.0554254, -0.0856021, 0.0821258
2: -0.0622640, 0.0955307, -0.0789161, 0.1109374, -0.1732014, 0.1744468
3: -0.0332325, 0.0495364, -0.0402230, 0.0574996, -0.0907322, 0.0897594
4: -0.0597730, 0.1052151, -0.0770320, 0.1258029, -0.1855760, 0.1822471

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886413, upper bound: 0.0886988
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36

Time for candidate selection: 2.35 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 33
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 23
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 23
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 34
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 26
type: A, layer: 5, pos: 49
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 1
type: A, layer: 5, pos: 16
type: A, layer: 5, pos: 14
type: A, layer: 5, pos: 42

Time for candidate selection: 5.98 seconds

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0865784, upper bound: 0.0874344
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882805, upper bound: 0.0884108
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0237667, 0.0206726, -0.0444393, 0.0444393
1: -0.0301767, 0.0481623, -0.0301767, 0.0481623, -0.0783390, 0.0783390
2: -0.0622640, 0.0955307, -0.0622640, 0.0955307, -0.1577947, 0.1577947
3: -0.0332325, 0.0495364, -0.0332325, 0.0495364, -0.0827689, 0.0827689
4: -0.0597730, 0.1052151, -0.0597730, 0.1052151, -0.1649881, 0.1649881

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11

Time for candidate selection: 1.98 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 49
type: B, layer: 5, pos: 49
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 33
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 23
type: B, layer: 5, pos: 23
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 14
type: B, layer: 5, pos: 14
type: A, layer: 5, pos: 1
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 16
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 42
type: B, layer: 5, pos: 42

Time for candidate selection: 5.62 seconds

### Candidate
type: A, layer: 5, pos: 49

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 49

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0875554, upper bound: 0.0872320
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882805, upper bound: 0.0884779
time: 0.40 seconds

## BFS IS instance: IS_A1_B1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0322836, 0.0334207, -0.0571874, 0.0529562
1: -0.0301767, 0.0481623, -0.0396060, 0.0710703, -0.1012470, 0.0877684
2: -0.0622640, 0.0955307, -0.0933639, 0.1287424, -0.1910064, 0.1888946
3: -0.0332325, 0.0495364, -0.0451630, 0.0779622, -0.1111947, 0.0946994
4: -0.0597730, 0.1052151, -0.0942796, 0.1455511, -0.2053241, 0.1994947

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 11

Time for candidate selection: 1.97 seconds

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 33
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 15
type: A, layer: 5, pos: 23
type: B, layer: 5, pos: 23
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 1
type: A, layer: 5, pos: 49
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 16
type: B, layer: 5, pos: 26
type: A, layer: 5, pos: 14
type: A, layer: 5, pos: 42

Time for candidate selection: 5.60 seconds

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0867950, upper bound: 0.0876077
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_A1_B2_B2_B2

### Relational analysis result of IS_A1_B1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882805, upper bound: 0.0884779
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0322836, 0.0334207, -0.0189038, 0.0158268, -0.0481104, 0.0523245
1: -0.0396060, 0.0710703, -0.0248756, 0.0358514, -0.0754574, 0.0959459
2: -0.0933639, 0.1287424, -0.0468024, 0.0796161, -0.1729800, 0.1755448
3: -0.0451630, 0.0779622, -0.0280160, 0.0339445, -0.0791075, 0.1059782
4: -0.0942796, 0.1455511, -0.0429375, 0.0872777, -0.1815574, 0.1884885

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 21

Time for candidate selection: 1.89 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 33
type: B, layer: 5, pos: 26
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 26
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 23
type: A, layer: 5, pos: 34
type: A, layer: 5, pos: 1
type: B, layer: 5, pos: 34
type: B, layer: 5, pos: 16
type: B, layer: 5, pos: 23
type: A, layer: 5, pos: 16
type: B, layer: 5, pos: 49
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 28
type: B, layer: 5, pos: 42

Time for candidate selection: 5.27 seconds

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 26

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_B1

### Relational analysis result of IS_A1_B1_A2_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0870365, upper bound: 0.0872818
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_B1_B2

### Relational analysis result of IS_A1_B1_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885197, upper bound: 0.0883044
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0322836, 0.0334207, -0.0237667, 0.0206726, -0.0529562, 0.0571874
1: -0.0396060, 0.0710703, -0.0301767, 0.0481623, -0.0877684, 0.1012470
2: -0.0933639, 0.1287424, -0.0622640, 0.0955307, -0.1888946, 0.1910064
3: -0.0451630, 0.0779622, -0.0332325, 0.0495364, -0.0946994, 0.1111947
4: -0.0942796, 0.1455511, -0.0597730, 0.1052151, -0.1994947, 0.2053241

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 11

Time for candidate selection: 1.87 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 33
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 15
type: B, layer: 5, pos: 23
type: A, layer: 5, pos: 23
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 16
type: B, layer: 5, pos: 1
type: B, layer: 5, pos: 49
type: A, layer: 5, pos: 1
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 26
type: B, layer: 5, pos: 14
type: B, layer: 5, pos: 42

Time for candidate selection: 5.34 seconds

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0877663, upper bound: 0.0867878
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885197, upper bound: 0.0883209
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0322836, 0.0334207, -0.0274795, 0.0253483, -0.0576319, 0.0609002
1: -0.0396060, 0.0710703, -0.0339634, 0.0554254, -0.0950314, 0.1050337
2: -0.0933639, 0.1287424, -0.0789161, 0.1109374, -0.2043013, 0.2076585
3: -0.0451630, 0.0779622, -0.0402230, 0.0574996, -0.1026626, 0.1181852
4: -0.0942796, 0.1455511, -0.0770320, 0.1258029, -0.2200826, 0.2225831

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 11

Time for candidate selection: 1.80 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 33
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 26
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 15
type: A, layer: 5, pos: 23
type: A, layer: 5, pos: 26
type: B, layer: 5, pos: 23
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 28
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 16
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 16
type: A, layer: 5, pos: 1

Time for candidate selection: 5.14 seconds

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_B1

### Relational analysis result of IS_A1_B1_A2_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0864003, upper bound: 0.0870174
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_B1_B2

### Relational analysis result of IS_A1_B1_A2_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880804, upper bound: 0.0879938
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0322836, 0.0334207, -0.0324717, 0.0338809, -0.0661644, 0.0658924
1: -0.0396060, 0.0710703, -0.0397940, 0.0717223, -0.1113284, 0.1108642
2: -0.0933639, 0.1287424, -0.0939188, 0.1294339, -0.2227978, 0.2226612
3: -0.0451630, 0.0779622, -0.0453307, 0.0787820, -0.1239450, 0.1232929
4: -0.0942796, 0.1455511, -0.0948957, 0.1463000, -0.2405796, 0.2404468

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11

Time for candidate selection: 1.77 seconds

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 33
type: A, layer: 5, pos: 33
type: A, layer: 5, pos: 26
type: B, layer: 5, pos: 26
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 21
type: B, layer: 5, pos: 23
type: A, layer: 5, pos: 23
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 16
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 1
type: B, layer: 5, pos: 1
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 28

Time for candidate selection: 5.31 seconds

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 26

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 26

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_A2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0872356, upper bound: 0.0864774
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_A2_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880804, upper bound: 0.0880275
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0271122, 0.0295338, -0.0242503, 0.0299525, -0.0570647, 0.0537841
1: -0.0338762, 0.0683478, -0.0366645, 0.0839245, -0.1178007, 0.1050123
2: -0.0775269, 0.1190995, -0.0667591, 0.1147048, -0.1922317, 0.1858586
3: -0.0413828, 0.0783211, -0.0451630, 0.1024292, -0.1438120, 0.1234841
4: -0.0817343, 0.1358244, -0.0822306, 0.1366947, -0.2184290, 0.2180549

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886952, upper bound: 0.0891720
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887542, upper bound: 0.0889950
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0280932, 0.0294513, -0.0242503, 0.0299525, -0.0580457, 0.0537016
1: -0.0348121, 0.0680492, -0.0366645, 0.0839245, -0.1187366, 0.1047137
2: -0.0808951, 0.1180537, -0.0667591, 0.1147048, -0.1955999, 0.1848128
3: -0.0416559, 0.0777593, -0.0451630, 0.1024292, -0.1440851, 0.1229223
4: -0.0845517, 0.1341558, -0.0822306, 0.1366947, -0.2212464, 0.2163864

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886126, upper bound: 0.0891720
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887542, upper bound: 0.0889950
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0189038, 0.0158268, -0.0291563, 0.0336461, -0.0525499, 0.0449831
1: -0.0248756, 0.0358514, -0.0408317, 0.0895138, -0.1143894, 0.0766831
2: -0.0468024, 0.0796161, -0.0816707, 0.1277841, -0.1745865, 0.1612867
3: -0.0280160, 0.0339445, -0.0488912, 0.1076730, -0.1356890, 0.0828357
4: -0.0429375, 0.0872777, -0.0948198, 0.1511557, -0.1940931, 0.1820976

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886940, upper bound: 0.0891682
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885696, upper bound: 0.0891682
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0274795, 0.0253483, -0.0285387, 0.0320224, -0.0595019, 0.0538870
1: -0.0339634, 0.0554254, -0.0398431, 0.0858412, -0.1198046, 0.0952684
2: -0.0789161, 0.1109374, -0.0796818, 0.1244380, -0.2033541, 0.1906192
3: -0.0402230, 0.0574996, -0.0478556, 0.1029174, -0.1431404, 0.1053552
4: -0.0770320, 0.1258029, -0.0919876, 0.1470573, -0.2240893, 0.2177905

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887542, upper bound: 0.0889951
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887542, upper bound: 0.0889950
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0271122, 0.0295338, -0.0295995, 0.0398316, -0.0669438, 0.0591333
1: -0.0338762, 0.0683478, -0.0433962, 0.1045852, -0.1384614, 0.1117440
2: -0.0775269, 0.1190995, -0.0837515, 0.1380744, -0.2156013, 0.2028509
3: -0.0413828, 0.0783211, -0.0519528, 0.1288300, -0.1702127, 0.1302739
4: -0.0817343, 0.1358244, -0.1026442, 0.1638758, -0.2456101, 0.2384686

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886866, upper bound: 0.0891797
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887520, upper bound: 0.0890131
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0280932, 0.0294513, -0.0295995, 0.0398316, -0.0679248, 0.0590508
1: -0.0348121, 0.0680492, -0.0433962, 0.1045852, -0.1393973, 0.1114454
2: -0.0808951, 0.1180537, -0.0837515, 0.1380744, -0.2189695, 0.2018051
3: -0.0416559, 0.0777593, -0.0519528, 0.1288300, -0.1704859, 0.1297121
4: -0.0845517, 0.1341558, -0.1026442, 0.1638758, -0.2484275, 0.2368000

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886866, upper bound: 0.0891797
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887520, upper bound: 0.0890131
time: 0.30 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0189038, 0.0158268, -0.0337088, 0.0468724, -0.0657761, 0.0495356
1: -0.0248756, 0.0358514, -0.0466997, 0.1138791, -0.1387547, 0.0825511
2: -0.0468024, 0.0796161, -0.0970538, 0.1534397, -0.2002421, 0.1766699
3: -0.0280160, 0.0339445, -0.0549959, 0.1400864, -0.1681024, 0.0889404
4: -0.0429375, 0.0872777, -0.1155892, 0.1808250, -0.2237624, 0.2028669

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885969, upper bound: 0.0891537
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885969, upper bound: 0.0891537
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0274795, 0.0253483, -0.0330995, 0.0447402, -0.0722197, 0.0584477
1: -0.0339634, 0.0554254, -0.0457167, 0.1096071, -0.1435705, 0.1011420
2: -0.0789161, 0.1109374, -0.0951445, 0.1497803, -0.2286964, 0.2060819
3: -0.0402230, 0.0574996, -0.0540131, 0.1343300, -0.1745530, 0.1115127
4: -0.0770320, 0.1258029, -0.1127598, 0.1764415, -0.2534735, 0.2385627

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887073, upper bound: 0.0890131
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887073, upper bound: 0.0890131
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0220832, 0.0217901, -0.0455568, 0.0427558
1: -0.0301767, 0.0481623, -0.0302110, 0.0556888, -0.0858655, 0.0783733
2: -0.0622640, 0.0955307, -0.0553987, 0.0898389, -0.1521029, 0.1509294
3: -0.0332325, 0.0495364, -0.0349907, 0.0610559, -0.0942885, 0.0845271
4: -0.0597730, 0.1052151, -0.0579685, 0.1031747, -0.1629478, 0.1631836

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888219, upper bound: 0.0892677
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 16

Time for candidate selection: 2.21 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884387, upper bound: 0.0891018
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887698, upper bound: 0.0892247
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888269, upper bound: 0.0892696
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 49
type: B, layer: 5, pos: 49
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 33
type: A, layer: 5, pos: 21
type: A, layer: 5, pos: 23
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 23
type: B, layer: 5, pos: 24
type: B, layer: 5, pos: 15
type: B, layer: 5, pos: 26
type: B, layer: 5, pos: 34
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 1
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 14
type: A, layer: 5, pos: 16
type: A, layer: 5, pos: 42
type: B, layer: 5, pos: 48
type: B, layer: 5, pos: 2

Time for candidate selection: 8.03 seconds

### Candidate
type: A, layer: 5, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0866244, upper bound: 0.0874386
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884270, upper bound: 0.0889232
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0267545, 0.0292090, -0.0529757, 0.0474271
1: -0.0301767, 0.0481623, -0.0378200, 0.0736853, -0.1038620, 0.0859823
2: -0.0622640, 0.0955307, -0.0743024, 0.1153906, -0.1776547, 0.1698331
3: -0.0332325, 0.0495364, -0.0447874, 0.0836997, -0.1169323, 0.0943238
4: -0.0597730, 0.1052151, -0.0820796, 0.1356305, -0.1954035, 0.1872947

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888219, upper bound: 0.0892677
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 16

Time for candidate selection: 2.22 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884387, upper bound: 0.0891018
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887698, upper bound: 0.0892247
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888269, upper bound: 0.0892696
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 33
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 26
type: B, layer: 5, pos: 24
type: B, layer: 5, pos: 16
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 23
type: A, layer: 5, pos: 49
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 1
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 28
type: B, layer: 5, pos: 23
type: A, layer: 5, pos: 28
type: A, layer: 5, pos: 16
type: A, layer: 5, pos: 14
type: A, layer: 5, pos: 42
type: B, layer: 5, pos: 48
type: B, layer: 5, pos: 2

Time for candidate selection: 8.11 seconds

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0866244, upper bound: 0.0874386
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884270, upper bound: 0.0889232
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0322836, 0.0334207, -0.0220832, 0.0217901, -0.0540737, 0.0555038
1: -0.0396060, 0.0710703, -0.0302110, 0.0556888, -0.0952948, 0.1012812
2: -0.0933639, 0.1287424, -0.0553987, 0.0898389, -0.1832028, 0.1841411
3: -0.0451630, 0.0779622, -0.0349907, 0.0610559, -0.1062189, 0.1129529
4: -0.0942796, 0.1455511, -0.0579685, 0.1031747, -0.1974544, 0.2035196

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886306, upper bound: 0.0890183
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 16

Time for candidate selection: 2.31 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885770, upper bound: 0.0889951
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882770, upper bound: 0.0888526
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886241, upper bound: 0.0890155
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 26
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 26
type: A, layer: 5, pos: 21
type: A, layer: 5, pos: 23
type: A, layer: 5, pos: 15
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 23
type: B, layer: 5, pos: 24
type: B, layer: 5, pos: 15
type: B, layer: 5, pos: 28
type: B, layer: 5, pos: 1
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 1
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 16
type: B, layer: 5, pos: 49
type: B, layer: 5, pos: 48
type: B, layer: 5, pos: 2

Time for candidate selection: 8.48 seconds

### Candidate
type: B, layer: 5, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0863060, upper bound: 0.0870216
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881650, upper bound: 0.0886880
time: 0.38 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0322836, 0.0334207, -0.0267545, 0.0292090, -0.0614925, 0.0601751
1: -0.0396060, 0.0710703, -0.0378200, 0.0736853, -0.1132914, 0.1088902
2: -0.0933639, 0.1287424, -0.0743024, 0.1153906, -0.2087545, 0.2030448
3: -0.0451630, 0.0779622, -0.0447874, 0.0836997, -0.1288627, 0.1227496
4: -0.0942796, 0.1455511, -0.0820796, 0.1356305, -0.2299101, 0.2276307

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886306, upper bound: 0.0890427
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 35

Time for candidate selection: 2.37 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882770, upper bound: 0.0888770
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885770, upper bound: 0.0890190
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 35

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886241, upper bound: 0.0890404
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 26
type: B, layer: 5, pos: 33
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 26
type: A, layer: 5, pos: 21
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 15
type: B, layer: 5, pos: 24
type: B, layer: 5, pos: 16
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 23
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 1
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 23
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 16
type: B, layer: 5, pos: 48
type: B, layer: 5, pos: 2

Time for candidate selection: 8.49 seconds

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0863060, upper bound: 0.0870216
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881650, upper bound: 0.0887285
time: 0.39 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0267169, 0.0278110, -0.0515777, 0.0473896
1: -0.0301767, 0.0481623, -0.0364216, 0.0731856, -0.1033623, 0.0845839
2: -0.0622640, 0.0955307, -0.0707564, 0.1097736, -0.1720376, 0.1662871
3: -0.0332325, 0.0495364, -0.0413534, 0.0851872, -0.1184198, 0.0908897
4: -0.0597730, 0.1052151, -0.0782428, 0.1268935, -0.1866666, 0.1834579

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887921, upper bound: 0.0892537
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 16

Time for candidate selection: 2.32 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0866107, upper bound: 0.0878297
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888319, upper bound: 0.0892734
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0328860, 0.0417665, -0.0655332, 0.0535587
1: -0.0301767, 0.0481623, -0.0459161, 0.0974250, -0.1276017, 0.0940784
2: -0.0622640, 0.0955307, -0.0932824, 0.1424614, -0.2047253, 0.1888131
3: -0.0332325, 0.0495364, -0.0520038, 0.1150784, -0.1483110, 0.1015402
4: -0.0597730, 0.1052151, -0.1057796, 0.1665460, -0.2263190, 0.2109947

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887921, upper bound: 0.0892537
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 35

Time for candidate selection: 2.33 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.1000000, mid=0.1000000, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 1) starts
Candidate diff: 0.0500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0897462, upper bound: 0.0898276
time: 0.32 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899608, upper bound: 0.0899608
time: 0.33 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.80 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -0.0897462, upper bound: 0.0898276
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -0.0899608, upper bound: 0.0899608

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0355980, 0.0491149, -0.0361746, 0.0480235, -0.0836215, 0.0852895
1: -0.0454062, 0.1038414, -0.0508857, 0.1093248, -0.1547310, 0.1547271
2: -0.1046092, 0.1549752, -0.1042037, 0.1570919, -0.2617012, 0.2591788
3: -0.0520488, 0.1242788, -0.0573016, 0.1296847, -0.1817336, 0.1815804
4: -0.1149560, 0.1775723, -0.1169854, 0.1851801, -0.3001361, 0.2945577

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896813, upper bound: 0.0896813
time: 0.32 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896813, upper bound: 0.0898276
time: 0.32 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0383316, 0.0603406, -0.0390887, 0.0634215, -0.1017531, 0.0994293
1: -0.0543237, 0.1360877, -0.0557757, 0.1414140, -0.1957377, 0.1918634
2: -0.1110429, 0.1767848, -0.1134923, 0.1820314, -0.2930743, 0.2902771
3: -0.0620208, 0.1679080, -0.0636387, 0.1747649, -0.2367857, 0.2315467
4: -0.1327446, 0.2087907, -0.1363116, 0.2151742, -0.3479187, 0.3451021

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898276, upper bound: 0.0897462
time: 0.32 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898276, upper bound: 0.0899608
time: 0.34 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.39 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 0, lower bound: -0.0896813, upper bound: 0.0896813
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 0, lower bound: -0.0896813, upper bound: 0.0898276
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 0, lower bound: -0.0898276, upper bound: 0.0897462
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.39
Output dim: 0, lower bound: -0.0898276, upper bound: 0.0899608

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0355980, 0.0491149, -0.0355980, 0.0491149, -0.0847130, 0.0847130
1: -0.0454062, 0.1038414, -0.0454062, 0.1038414, -0.1492476, 0.1492476
2: -0.1046092, 0.1549752, -0.1046092, 0.1549752, -0.2595844, 0.2595844
3: -0.0520488, 0.1242788, -0.0520488, 0.1242788, -0.1763276, 0.1763276
4: -0.1149560, 0.1775723, -0.1149560, 0.1775723, -0.2925282, 0.2925282

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896273, upper bound: 0.0896811
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896813, upper bound: 0.0896813
time: 0.32 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0355980, 0.0491149, -0.0378599, 0.0574707, -0.0930687, 0.0869748
1: -0.0454062, 0.1038414, -0.0537177, 0.1317393, -0.1771455, 0.1575590
2: -0.1046092, 0.1549752, -0.1096742, 0.1730022, -0.2776114, 0.2646493
3: -0.0520488, 0.1242788, -0.0613368, 0.1622476, -0.2142964, 0.1856156
4: -0.1149560, 0.1775723, -0.1303308, 0.2046323, -0.3195883, 0.3079031

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896273, upper bound: 0.0896811
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896813, upper bound: 0.0896813
time: 0.32 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0383316, 0.0603406, -0.0355980, 0.0491149, -0.0874465, 0.0959386
1: -0.0543237, 0.1360877, -0.0454062, 0.1038414, -0.1581650, 0.1814938
2: -0.1110429, 0.1767848, -0.1046092, 0.1549752, -0.2660180, 0.2813941
3: -0.0620208, 0.1679080, -0.0520488, 0.1242788, -0.1862996, 0.2199568
4: -0.1327446, 0.2087907, -0.1149560, 0.1775723, -0.3103168, 0.3237466

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898263, upper bound: 0.0896900
time: 0.34 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898276, upper bound: 0.0897453
time: 0.32 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0383316, 0.0603406, -0.0383316, 0.0603406, -0.0986722, 0.0986722
1: -0.0543237, 0.1360877, -0.0543237, 0.1360877, -0.1904113, 0.1904114
2: -0.1110429, 0.1767848, -0.1110429, 0.1767848, -0.2878277, 0.2878277
3: -0.0620208, 0.1679080, -0.0620208, 0.1679080, -0.2299288, 0.2299288
4: -0.1327446, 0.2087907, -0.1327446, 0.2087907, -0.3415352, 0.3415352

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898267, upper bound: 0.0898844
time: 0.32 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898276, upper bound: 0.0897453
time: 0.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.96 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -0.0896273, upper bound: 0.0896811
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -0.0896813, upper bound: 0.0896813
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -0.0896273, upper bound: 0.0896811
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -0.0896813, upper bound: 0.0896813
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -0.0898263, upper bound: 0.0896900
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -0.0898276, upper bound: 0.0897453
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -0.0898267, upper bound: 0.0898844
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -0.0898276, upper bound: 0.0897453

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0355980, 0.0491149, -0.0780348, 0.0672179
1: -0.0364605, 0.0725038, -0.0454062, 0.1038414, -0.1403019, 0.1179100
2: -0.0836895, 0.1224094, -0.1046092, 0.1549752, -0.2386647, 0.2270187
3: -0.0432477, 0.0836505, -0.0520488, 0.1242788, -0.1675265, 0.1356993
4: -0.0884080, 0.1397506, -0.1149560, 0.1775723, -0.2659802, 0.2547066

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896271, upper bound: 0.0896271
time: 0.29 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896271, upper bound: 0.0896811
time: 0.30 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0341963, 0.0451829, -0.0355980, 0.0491149, -0.0833112, 0.0807809
1: -0.0427969, 0.0966445, -0.0454062, 0.1038414, -0.1466383, 0.1420507
2: -0.1001576, 0.1478489, -0.1046092, 0.1549752, -0.2551327, 0.2524582
3: -0.0493405, 0.1152938, -0.0520488, 0.1242788, -0.1736193, 0.1673426
4: -0.1093097, 0.1684246, -0.1149560, 0.1775723, -0.2868820, 0.2833806

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896811, upper bound: 0.0896273
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896811, upper bound: 0.0896813
time: 0.32 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0378599, 0.0574707, -0.0863906, 0.0694798
1: -0.0364605, 0.0725038, -0.0537177, 0.1317393, -0.1681999, 0.1262214
2: -0.0836895, 0.1224094, -0.1096742, 0.1730022, -0.2566917, 0.2320836
3: -0.0432477, 0.0836505, -0.0613368, 0.1622476, -0.2054953, 0.1449873
4: -0.0884080, 0.1397506, -0.1303308, 0.2046323, -0.2930403, 0.2700814

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887379, upper bound: 0.0889640
time: 0.29 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887906, upper bound: 0.0890124
time: 0.31 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0341963, 0.0451829, -0.0378599, 0.0574707, -0.0916670, 0.0830428
1: -0.0427969, 0.0966445, -0.0537177, 0.1317393, -0.1745363, 0.1503622
2: -0.1001576, 0.1478489, -0.1096742, 0.1730022, -0.2731597, 0.2575230
3: -0.0493405, 0.1152938, -0.0613368, 0.1622476, -0.2115881, 0.1766306
4: -0.1093097, 0.1684246, -0.1303308, 0.2046323, -0.3139420, 0.2987553

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0897293, upper bound: 0.0898267
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0897293, upper bound: 0.0898276
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0383316, 0.0603406, -0.0289199, 0.0316199, -0.0699515, 0.0892605
1: -0.0543237, 0.1360877, -0.0364605, 0.0725038, -0.1268275, 0.1725482
2: -0.1110429, 0.1767848, -0.0836895, 0.1224094, -0.2334523, 0.2604744
3: -0.0620208, 0.1679080, -0.0432477, 0.0836505, -0.1456713, 0.2111557
4: -0.1327446, 0.2087907, -0.0884080, 0.1397506, -0.2724952, 0.2971986

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889640, upper bound: 0.0887379
time: 0.34 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890124, upper bound: 0.0887906
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0383316, 0.0603406, -0.0341963, 0.0451829, -0.0835145, 0.0945368
1: -0.0543237, 0.1360877, -0.0427969, 0.0966445, -0.1509682, 0.1788846
2: -0.1110429, 0.1767848, -0.1001576, 0.1478489, -0.2588917, 0.2769423
3: -0.0620208, 0.1679080, -0.0493405, 0.1152938, -0.1773146, 0.2172485
4: -0.1327446, 0.2087907, -0.1093097, 0.1684246, -0.3011691, 0.3181004

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898267, upper bound: 0.0897293
time: 0.33 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0898267, upper bound: 0.0897453
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0303880, 0.0381159, -0.0383316, 0.0603406, -0.0907286, 0.0764475
1: -0.0436567, 0.0986741, -0.0543237, 0.1360877, -0.1797444, 0.1529978
2: -0.0858032, 0.1358688, -0.1110429, 0.1767848, -0.2625880, 0.2469117
3: -0.0513058, 0.1197162, -0.0620208, 0.1679080, -0.2192139, 0.1817370
4: -0.1010720, 0.1610686, -0.1327446, 0.2087907, -0.3098626, 0.2938131

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893540, upper bound: 0.0882400
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885870, upper bound: 0.0881785
time: 0.32 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0352700, 0.0520305, -0.0383316, 0.0603406, -0.0956105, 0.0903621
1: -0.0497746, 0.1236877, -0.0543237, 0.1360877, -0.1858622, 0.1780114
2: -0.1017607, 0.1623120, -0.1110429, 0.1767848, -0.2785456, 0.2733549
3: -0.0576897, 0.1530293, -0.0620208, 0.1679080, -0.2255976, 0.2150501
4: -0.1224101, 0.1915598, -0.1327446, 0.2087907, -0.3312007, 0.3243043

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896430, upper bound: 0.0888528
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
time: 0.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.61 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0896271, upper bound: 0.0896271
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0896271, upper bound: 0.0896811
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0896811, upper bound: 0.0896273
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0896811, upper bound: 0.0896813
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0887379, upper bound: 0.0889640
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0887906, upper bound: 0.0890124
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0897293, upper bound: 0.0898267
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0897293, upper bound: 0.0898276
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0889640, upper bound: 0.0887379
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0890124, upper bound: 0.0887906
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0898267, upper bound: 0.0897293
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0898267, upper bound: 0.0897453
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0893540, upper bound: 0.0882400
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0885870, upper bound: 0.0881785
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0896430, upper bound: 0.0888528
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.61
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0289199, 0.0316199, -0.0605398, 0.0605398
1: -0.0364605, 0.0725038, -0.0364605, 0.0725038, -0.1089643, 0.1089643
2: -0.0836895, 0.1224094, -0.0836895, 0.1224094, -0.2060990, 0.2060990
3: -0.0432477, 0.0836505, -0.0432477, 0.0836505, -0.1268981, 0.1268981
4: -0.0884080, 0.1397506, -0.0884080, 0.1397506, -0.2281586, 0.2281586

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894284, upper bound: 0.0895061
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887628, upper bound: 0.0886505
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885535, upper bound: 0.0885538
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0341963, 0.0451829, -0.0741027, 0.0658161
1: -0.0364605, 0.0725038, -0.0427969, 0.0966445, -0.1331051, 0.1153007
2: -0.0836895, 0.1224094, -0.1001576, 0.1478489, -0.2315385, 0.2225670
3: -0.0432477, 0.0836505, -0.0493405, 0.1152938, -0.1585415, 0.1329910
4: -0.0884080, 0.1397506, -0.1093097, 0.1684246, -0.2568326, 0.2490603

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887628, upper bound: 0.0887371
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885535, upper bound: 0.0886361
time: 0.30 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0341963, 0.0451829, -0.0289199, 0.0316199, -0.0658161, 0.0741027
1: -0.0427969, 0.0966445, -0.0364605, 0.0725038, -0.1153007, 0.1331051
2: -0.1001576, 0.1478489, -0.0836895, 0.1224094, -0.2225670, 0.2315385
3: -0.0493405, 0.1152938, -0.0432477, 0.0836505, -0.1329910, 0.1585415
4: -0.1093097, 0.1684246, -0.0884080, 0.1397506, -0.2490603, 0.2568325

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886500, upper bound: 0.0887628
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886358, upper bound: 0.0885535
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0341963, 0.0451829, -0.0341963, 0.0451829, -0.0793791, 0.0793791
1: -0.0427969, 0.0966445, -0.0427969, 0.0966445, -0.1394415, 0.1394415
2: -0.1001576, 0.1478489, -0.1001576, 0.1478489, -0.2480065, 0.2480064
3: -0.0493405, 0.1152938, -0.0493405, 0.1152938, -0.1646343, 0.1646343
4: -0.1093097, 0.1684246, -0.1093097, 0.1684246, -0.2777342, 0.2777343

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886500, upper bound: 0.0887726
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886358, upper bound: 0.0885564
time: 0.31 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0189038, 0.0158268, -0.0369843, 0.0537019, -0.0726057, 0.0528111
1: -0.0248756, 0.0358514, -0.0523166, 0.1249076, -0.1497832, 0.0881680
2: -0.0468024, 0.0796161, -0.1064229, 0.1658513, -0.2126537, 0.1860390
3: -0.0280160, 0.0339445, -0.0594187, 0.1530832, -0.1810991, 0.0933632
4: -0.0429375, 0.0872777, -0.1251003, 0.1958140, -0.2387515, 0.2123781

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884727, upper bound: 0.0888586
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887379, upper bound: 0.0889554
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887379, upper bound: 0.0889640
time: 0.31 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0274795, 0.0253483, -0.0347391, 0.0477613, -0.0752408, 0.0600874
1: -0.0339634, 0.0554254, -0.0488479, 0.1117781, -0.1457415, 0.1042733
2: -0.0789161, 0.1109374, -0.0996932, 0.1560862, -0.2350023, 0.2106306
3: -0.0402230, 0.0574996, -0.0564002, 0.1354525, -0.1756755, 0.1138998
4: -0.0770320, 0.1258029, -0.1158940, 0.1844381, -0.2614702, 0.2416969

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885491, upper bound: 0.0889639
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887906, upper bound: 0.0889769
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887906, upper bound: 0.0890124
time: 0.31 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0341963, 0.0451829, -0.0300616, 0.0366620, -0.0708582, 0.0752444
1: -0.0427969, 0.0966445, -0.0432292, 0.0955959, -0.1383928, 0.1398737
2: -0.1001576, 0.1478489, -0.0848061, 0.1335995, -0.2337570, 0.2326550
3: -0.0493405, 0.1152938, -0.0507821, 0.1153900, -0.1647305, 0.1660759
4: -0.1093097, 0.1684246, -0.0992707, 0.1584232, -0.2677329, 0.2676951

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896808, upper bound: 0.0898192
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888632, upper bound: 0.0889554
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889098, upper bound: 0.0889766
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0341963, 0.0451829, -0.0347878, 0.0496648, -0.0838611, 0.0799706
1: -0.0427969, 0.0966445, -0.0491802, 0.1195409, -0.1623378, 0.1458248
2: -0.1001576, 0.1478489, -0.1004114, 0.1591337, -0.2592912, 0.2482604
3: -0.0493405, 0.1152938, -0.0569754, 0.1474385, -0.1967790, 0.1722692
4: -0.1093097, 0.1684246, -0.1200407, 0.1879030, -0.2972127, 0.2884652

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888632, upper bound: 0.0889560
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889098, upper bound: 0.0889797
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0373791, 0.0560596, -0.0189038, 0.0158268, -0.0532059, 0.0749634
1: -0.0528793, 0.1287078, -0.0248756, 0.0358514, -0.0887307, 0.1535834
2: -0.1076983, 0.1690182, -0.0468024, 0.0796161, -0.1873143, 0.2158206
3: -0.0600504, 0.1581047, -0.0280160, 0.0339445, -0.0939949, 0.1861206
4: -0.1273126, 0.1993461, -0.0429375, 0.0872777, -0.2145903, 0.2422835

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888586, upper bound: 0.0884727
time: 0.33 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889554, upper bound: 0.0887379
time: 0.32 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889554, upper bound: 0.0887379
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0351993, 0.0495232, -0.0274795, 0.0253483, -0.0605475, 0.0770027
1: -0.0493625, 0.1146552, -0.0339634, 0.0554254, -0.1047879, 0.1486186
2: -0.1010823, 0.1586219, -0.0789161, 0.1109374, -0.2120197, 0.2375380
3: -0.0570570, 0.1391938, -0.0402230, 0.0574996, -0.1145566, 0.1794168
4: -0.1179667, 0.1873384, -0.0770320, 0.1258029, -0.2437696, 0.2643705

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889639, upper bound: 0.0885491
time: 0.33 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889769, upper bound: 0.0887906
time: 0.34 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889769, upper bound: 0.0887906
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0303880, 0.0381159, -0.0341963, 0.0451829, -0.0755709, 0.0723121
1: -0.0436567, 0.0986741, -0.0427969, 0.0966445, -0.1403012, 0.1414710
2: -0.0858032, 0.1358688, -0.1001576, 0.1478489, -0.2336521, 0.2360264
3: -0.0513058, 0.1197162, -0.0493405, 0.1152938, -0.1665996, 0.1690567
4: -0.1010720, 0.1610686, -0.1093097, 0.1684246, -0.2694965, 0.2703783

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889554, upper bound: 0.0888632
time: 0.32 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889766, upper bound: 0.0889098
time: 0.31 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0352700, 0.0520305, -0.0341963, 0.0451829, -0.0804528, 0.0862267
1: -0.0497746, 0.1236877, -0.0427969, 0.0966445, -0.1464191, 0.1664846
2: -0.1017607, 0.1623120, -0.1001576, 0.1478489, -0.2496097, 0.2624695
3: -0.0576897, 0.1530293, -0.0493405, 0.1152938, -0.1729835, 0.2023698
4: -0.1224101, 0.1915598, -0.1093097, 0.1684246, -0.2908346, 0.3008695

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889554, upper bound: 0.0887386
time: 0.32 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889766, upper bound: 0.0887966
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0303880, 0.0381159, -0.0358490, 0.0487337, -0.0791218, 0.0739649
1: -0.0436567, 0.0986741, -0.0507856, 0.1154432, -0.1590999, 0.1494596
2: -0.0858032, 0.1358688, -0.1024819, 0.1582004, -0.2440036, 0.2383507
3: -0.0513058, 0.1197162, -0.0580275, 0.1404007, -0.1917065, 0.1777436
4: -0.1010720, 0.1610686, -0.1195223, 0.1871233, -0.2881952, 0.2805909

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885870, upper bound: 0.0881785
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885870, upper bound: 0.0881785
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0250051, 0.0305142, -0.0262249, 0.0476892, -0.0726942, 0.0567391
1: -0.0369577, 0.0859875, -0.0560731, 0.1461119, -0.1830696, 0.1420607
2: -0.0694120, 0.1166408, -0.0741902, 0.1451171, -0.2145290, 0.1908310
3: -0.0451636, 0.1057661, -0.0633892, 0.1894719, -0.2346355, 0.1691553
4: -0.0856764, 0.1383418, -0.1138687, 0.1800817, -0.2657581, 0.2522104

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885870, upper bound: 0.0881785
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885870, upper bound: 0.0881785
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0352700, 0.0520305, -0.0358490, 0.0487337, -0.0840037, 0.0878795
1: -0.0497746, 0.1236877, -0.0507856, 0.1154432, -0.1652177, 0.1744732
2: -0.1017607, 0.1623120, -0.1024819, 0.1582004, -0.2599612, 0.2647940
3: -0.0576897, 0.1530293, -0.0580275, 0.1404007, -0.1980903, 0.2110567
4: -0.1224101, 0.1915598, -0.1195223, 0.1871233, -0.3095332, 0.3110821

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0304444, 0.0416167, -0.0262249, 0.0476892, -0.0781335, 0.0678416
1: -0.0435608, 0.1074786, -0.0560731, 0.1461119, -0.1896727, 0.1635517
2: -0.0870023, 0.1416976, -0.0741902, 0.1451171, -0.2321194, 0.2158878
3: -0.0518835, 0.1331060, -0.0633892, 0.1894719, -0.2413554, 0.1964952
4: -0.1068231, 0.1674389, -0.1138687, 0.1800817, -0.2869048, 0.2813076

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
time: 0.36 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.69 seconds
IS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0887628, upper bound: 0.0886505
IS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0885535, upper bound: 0.0885538
IS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0887628, upper bound: 0.0887371
IS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0885535, upper bound: 0.0886361
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0886500, upper bound: 0.0887628
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0886358, upper bound: 0.0885535
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0886500, upper bound: 0.0887726
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0886358, upper bound: 0.0885564
IS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0887379, upper bound: 0.0889554
IS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0887379, upper bound: 0.0889640
IS_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0887906, upper bound: 0.0889769
IS_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0887906, upper bound: 0.0890124
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0888632, upper bound: 0.0889554
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0889098, upper bound: 0.0889766
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0888632, upper bound: 0.0889560
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0889098, upper bound: 0.0889797
IS_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0889554, upper bound: 0.0887379
IS_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0889554, upper bound: 0.0887379
IS_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0889769, upper bound: 0.0887906
IS_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0889769, upper bound: 0.0887906
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0889554, upper bound: 0.0888632
IS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0889766, upper bound: 0.0889098
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0889554, upper bound: 0.0887386
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0889766, upper bound: 0.0887966
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0885870, upper bound: 0.0881785
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0885870, upper bound: 0.0881785
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0885870, upper bound: 0.0881785
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0885870, upper bound: 0.0881785
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.69
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871

## BFS IS instance: IS_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0278064, 0.0288961, -0.0189038, 0.0158268, -0.0436331, 0.0477999
1: -0.0350639, 0.0669351, -0.0248756, 0.0358514, -0.0709153, 0.0918107
2: -0.0795793, 0.1169136, -0.0468024, 0.0796161, -0.1591953, 0.1637161
3: -0.0413456, 0.0759878, -0.0280160, 0.0339445, -0.0752901, 0.1040038
4: -0.0828490, 0.1328639, -0.0429375, 0.0872777, -0.1701268, 0.1758013

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885416, upper bound: 0.0886151
time: 0.29 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884416, upper bound: 0.0883745
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0270086, 0.0252807, -0.0274795, 0.0253483, -0.0523569, 0.0527602
1: -0.0336199, 0.0582085, -0.0339634, 0.0554254, -0.0890452, 0.0921719
2: -0.0777968, 0.1109209, -0.0789161, 0.1109374, -0.1887341, 0.1898370
3: -0.0399900, 0.0625573, -0.0402230, 0.0574996, -0.0974896, 0.1027803
4: -0.0778713, 0.1258984, -0.0770320, 0.1258029, -0.2036743, 0.2029304

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883070, upper bound: 0.0885059
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882591, upper bound: 0.0882591
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0278064, 0.0288961, -0.0237667, 0.0206726, -0.0484790, 0.0526628
1: -0.0350639, 0.0669351, -0.0301767, 0.0481623, -0.0832263, 0.0971118
2: -0.0795793, 0.1169136, -0.0622640, 0.0955307, -0.1751100, 0.1791777
3: -0.0413456, 0.0759878, -0.0332325, 0.0495364, -0.0908820, 0.1092204
4: -0.0828490, 0.1328639, -0.0597730, 0.1052151, -0.1880641, 0.1926369

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885416, upper bound: 0.0886974
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884127, upper bound: 0.0884950
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884127, upper bound: 0.0886361
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0270086, 0.0252807, -0.0322836, 0.0334207, -0.0604293, 0.0575643
1: -0.0336199, 0.0582085, -0.0396060, 0.0710703, -0.1046901, 0.0978145
2: -0.0777968, 0.1109209, -0.0933639, 0.1287424, -0.2065392, 0.2042848
3: -0.0399900, 0.0625573, -0.0451630, 0.0779622, -0.1179522, 0.1077202
4: -0.0778713, 0.1258984, -0.0942796, 0.1455511, -0.2234224, 0.2201780

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883067, upper bound: 0.0885881
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885535, upper bound: 0.0884950
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885535, upper bound: 0.0886361
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0278064, 0.0288961, -0.0526628, 0.0484790
1: -0.0301767, 0.0481623, -0.0350639, 0.0669351, -0.0971118, 0.0832263
2: -0.0622640, 0.0955307, -0.0795793, 0.1169136, -0.1791777, 0.1751100
3: -0.0332325, 0.0495364, -0.0413456, 0.0759878, -0.1092204, 0.0908820
4: -0.0597730, 0.1052151, -0.0828490, 0.1328639, -0.1926369, 0.1880641

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886974, upper bound: 0.0885416
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884127, upper bound: 0.0884127
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884127, upper bound: 0.0884127
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0322836, 0.0334207, -0.0270086, 0.0252807, -0.0575643, 0.0604293
1: -0.0396060, 0.0710703, -0.0336199, 0.0582085, -0.0978145, 0.1046901
2: -0.0933639, 0.1287424, -0.0777968, 0.1109209, -0.2042848, 0.2065392
3: -0.0451630, 0.0779622, -0.0399900, 0.0625573, -0.1077202, 0.1179522
4: -0.0942796, 0.1455511, -0.0778713, 0.1258984, -0.2201780, 0.2234224

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885881, upper bound: 0.0883067
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0885535
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0885535
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0330466, 0.0416848, -0.0654515, 0.0537193
1: -0.0301767, 0.0481623, -0.0412757, 0.0899316, -0.1201083, 0.0894380
2: -0.0622640, 0.0955307, -0.0958907, 0.1414468, -0.2037108, 0.1914213
3: -0.0332325, 0.0495364, -0.0472691, 0.1062783, -0.1395108, 0.0968055
4: -0.0597730, 0.1052151, -0.1033600, 0.1605898, -0.2203629, 0.2085751

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0884192
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0884192
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0322836, 0.0334207, -0.0321320, 0.0360750, -0.0683586, 0.0655526
1: -0.0396060, 0.0710703, -0.0391623, 0.0778116, -0.1174177, 0.1102326
2: -0.0933639, 0.1287424, -0.0934697, 0.1325096, -0.2258735, 0.2222120
3: -0.0451630, 0.0779622, -0.0454521, 0.0890502, -0.1342132, 0.1234143
4: -0.0942796, 0.1455511, -0.0968729, 0.1501236, -0.2444033, 0.2424240

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0885564
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0885564
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0189038, 0.0158268, -0.0292816, 0.0335811, -0.0524849, 0.0451083
1: -0.0248756, 0.0358514, -0.0418911, 0.0897424, -0.1146180, 0.0777425
2: -0.0468024, 0.0796161, -0.0817847, 0.1269688, -0.1737712, 0.1614007
3: -0.0280160, 0.0339445, -0.0489615, 0.1078001, -0.1358161, 0.0829060
4: -0.0429375, 0.0872777, -0.0948193, 0.1501165, -0.1930540, 0.1820971

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884727, upper bound: 0.0888497
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886327, upper bound: 0.0889554
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886327, upper bound: 0.0889554
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0189038, 0.0158268, -0.0338360, 0.0460914, -0.0649952, 0.0496628
1: -0.0248756, 0.0358514, -0.0478752, 0.1130114, -0.1378870, 0.0837266
2: -0.0468024, 0.0796161, -0.0971696, 0.1521198, -0.1989222, 0.1767857
3: -0.0280160, 0.0339445, -0.0550897, 0.1385629, -0.1665788, 0.0890342
4: -0.0429375, 0.0872777, -0.1148472, 0.1792814, -0.2222188, 0.2021250

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884068, upper bound: 0.0887167
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886327, upper bound: 0.0889640
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886327, upper bound: 0.0889640
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0274795, 0.0253483, -0.0271217, 0.0301507, -0.0576302, 0.0524700
1: -0.0339634, 0.0554254, -0.0384462, 0.0795161, -0.1134795, 0.0938715
2: -0.0789161, 0.1109374, -0.0754988, 0.1194209, -0.1983370, 0.1864362
3: -0.0402230, 0.0574996, -0.0461158, 0.0937034, -0.1339264, 0.1036154
4: -0.0770320, 0.1258029, -0.0859966, 0.1411357, -0.2181678, 0.2117995

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885491, upper bound: 0.0889288
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885475, upper bound: 0.0888740
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885475, upper bound: 0.0888740
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0274795, 0.0253483, -0.0321666, 0.0405496, -0.0680291, 0.0575149
1: -0.0339634, 0.0554254, -0.0449068, 0.1005462, -0.1345096, 0.1003322
2: -0.0789161, 0.1109374, -0.0919458, 0.1433038, -0.2222199, 0.2028832
3: -0.0402230, 0.0574996, -0.0524837, 0.1215736, -0.1617966, 0.1099833
4: -0.0770320, 0.1258029, -0.1070413, 0.1687967, -0.2458287, 0.2328443

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885491, upper bound: 0.0889639
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885475, upper bound: 0.0888740
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885475, upper bound: 0.0889614
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0292816, 0.0335811, -0.0573478, 0.0499542
1: -0.0301767, 0.0481623, -0.0418911, 0.0897424, -0.1199191, 0.0900534
2: -0.0622640, 0.0955307, -0.0817847, 0.1269688, -0.1892328, 0.1773154
3: -0.0332325, 0.0495364, -0.0489615, 0.1078001, -0.1410327, 0.0984978
4: -0.0597730, 0.1052151, -0.0948193, 0.1501165, -0.2098896, 0.2000344

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888692, upper bound: 0.0889508
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886295, upper bound: 0.0888141
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886295, upper bound: 0.0888830
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0322836, 0.0334207, -0.0271217, 0.0301507, -0.0624342, 0.0605424
1: -0.0396060, 0.0710703, -0.0384462, 0.0795161, -0.1191221, 0.1095164
2: -0.0933639, 0.1287424, -0.0754988, 0.1194209, -0.2127848, 0.2042412
3: -0.0451630, 0.0779622, -0.0461158, 0.0937034, -0.1388664, 0.1240780
4: -0.0942796, 0.1455511, -0.0859966, 0.1411357, -0.2354154, 0.2315477

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889168, upper bound: 0.0889578
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886295, upper bound: 0.0888737
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886295, upper bound: 0.0889766
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0338360, 0.0460914, -0.0698581, 0.0545086
1: -0.0301767, 0.0481623, -0.0478752, 0.1130114, -0.1431881, 0.0960376
2: -0.0622640, 0.0955307, -0.0971696, 0.1521198, -0.2143837, 0.1927003
3: -0.0332325, 0.0495364, -0.0550897, 0.1385629, -0.1717954, 0.1046261
4: -0.0597730, 0.1052151, -0.1148472, 0.1792814, -0.2390544, 0.2200623

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886415, upper bound: 0.0888164
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885475, upper bound: 0.0888835
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0322836, 0.0334207, -0.0321666, 0.0405496, -0.0728332, 0.0655873
1: -0.0396060, 0.0710703, -0.0449068, 0.1005462, -0.1401522, 0.1159771
2: -0.0933639, 0.1287424, -0.0919458, 0.1433038, -0.2366677, 0.2206882
3: -0.0451630, 0.0779622, -0.0524837, 0.1215736, -0.1667366, 0.1304458
4: -0.0942796, 0.1455511, -0.1070413, 0.1687967, -0.2630763, 0.2525924

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886415, upper bound: 0.0888810
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886415, upper bound: 0.0889797
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0295338, 0.0348291, -0.0189038, 0.0158268, -0.0453606, 0.0537329
1: -0.0422955, 0.0924248, -0.0248756, 0.0358514, -0.0781469, 0.1173003
2: -0.0826760, 0.1289816, -0.0468024, 0.0796161, -0.1622920, 0.1757840
3: -0.0494353, 0.1114590, -0.0280160, 0.0339445, -0.0833798, 0.1394750
4: -0.0963789, 0.1524827, -0.0429375, 0.0872777, -0.1836566, 0.1954202

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888497, upper bound: 0.0884727
time: 0.34 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889554, upper bound: 0.0886327
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889554, upper bound: 0.0887379
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0342729, 0.0482908, -0.0189038, 0.0158268, -0.0500997, 0.0671946
1: -0.0484162, 0.1167572, -0.0248756, 0.0358514, -0.0842676, 0.1416328
2: -0.0983613, 0.1550079, -0.0468024, 0.0796161, -0.1779774, 0.2018103
3: -0.0556786, 0.1436329, -0.0280160, 0.0339445, -0.0896231, 0.1716489
4: -0.1169464, 0.1825493, -0.0429375, 0.0872777, -0.2042241, 0.2254867

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888497, upper bound: 0.0884727
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889554, upper bound: 0.0886327
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889554, upper bound: 0.0887379
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0273881, 0.0305587, -0.0274795, 0.0253483, -0.0527364, 0.0580382
1: -0.0387803, 0.0807908, -0.0339634, 0.0554254, -0.0942057, 0.1147543
2: -0.0763814, 0.1205543, -0.0789161, 0.1109374, -0.1873188, 0.1994704
3: -0.0465575, 0.0956225, -0.0402230, 0.0574996, -0.1040572, 0.1358455
4: -0.0873188, 0.1425182, -0.0770320, 0.1258029, -0.2131217, 0.2195502

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889288, upper bound: 0.0885491
time: 0.34 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888141, upper bound: 0.0885475
time: 0.34 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888141, upper bound: 0.0887906
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0328059, 0.0423835, -0.0274795, 0.0253483, -0.0581542, 0.0698630
1: -0.0455719, 0.1034022, -0.0339634, 0.0554254, -0.1009973, 0.1373656
2: -0.0936920, 0.1460620, -0.0789161, 0.1109374, -0.2046294, 0.2249781
3: -0.0531151, 0.1252927, -0.0402230, 0.0574996, -0.1106148, 0.1655157
4: -0.1092365, 0.1718355, -0.0770320, 0.1258029, -0.2350394, 0.2488675

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889288, upper bound: 0.0885491
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888141, upper bound: 0.0885596
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888141, upper bound: 0.0887906
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0295338, 0.0348291, -0.0237667, 0.0206726, -0.0502064, 0.0585958
1: -0.0422955, 0.0924248, -0.0301767, 0.0481623, -0.0904579, 0.1226014
2: -0.0826760, 0.1289816, -0.0622640, 0.0955307, -0.1782067, 0.1912457
3: -0.0494353, 0.1114590, -0.0332325, 0.0495364, -0.0989717, 0.1446915
4: -0.0963789, 0.1524827, -0.0597730, 0.1052151, -0.2015940, 0.2122558

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889508, upper bound: 0.0888692
time: 0.36 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888141, upper bound: 0.0886295
time: 0.33 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888141, upper bound: 0.0888726
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0273881, 0.0305587, -0.0322836, 0.0334207, -0.0608088, 0.0628423
1: -0.0387803, 0.0807908, -0.0396060, 0.0710703, -0.1098506, 0.1203969
2: -0.0763814, 0.1205543, -0.0933639, 0.1287424, -0.2051238, 0.2139182
3: -0.0465575, 0.0956225, -0.0451630, 0.0779622, -0.1245197, 0.1407855
4: -0.0873188, 0.1425182, -0.0942796, 0.1455511, -0.2328698, 0.2367978

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889578, upper bound: 0.0889168
time: 0.36 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888737, upper bound: 0.0886295
time: 0.34 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888737, upper bound: 0.0886295
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0342729, 0.0482908, -0.0237667, 0.0206726, -0.0549455, 0.0720575
1: -0.0484162, 0.1167572, -0.0301767, 0.0481623, -0.0965785, 0.1469339
2: -0.0983613, 0.1550079, -0.0622640, 0.0955307, -0.1938920, 0.2172719
3: -0.0556786, 0.1436329, -0.0332325, 0.0495364, -0.1052150, 0.1768654
4: -0.1169464, 0.1825493, -0.0597730, 0.1052151, -0.2221615, 0.2423223

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888427, upper bound: 0.0885920
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888427, upper bound: 0.0887386
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0328059, 0.0423835, -0.0322836, 0.0334207, -0.0662266, 0.0746671
1: -0.0455719, 0.1034022, -0.0396060, 0.0710703, -0.1166421, 0.1430082
2: -0.0936920, 0.1460620, -0.0933639, 0.1287424, -0.2224344, 0.2394259
3: -0.0531151, 0.1252927, -0.0451630, 0.0779622, -0.1310773, 0.1704557
4: -0.1092365, 0.1718355, -0.0942796, 0.1455511, -0.2547876, 0.2661151

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889193, upper bound: 0.0885920
time: 0.34 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0889193, upper bound: 0.0885920
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0281139, 0.0310250, -0.0358490, 0.0487337, -0.0768476, 0.0668740
1: -0.0406024, 0.0818455, -0.0507856, 0.1154432, -0.1560455, 0.1326310
2: -0.0773617, 0.1205507, -0.1024819, 0.1582004, -0.2355621, 0.2230326
3: -0.0474901, 0.0966178, -0.0580275, 0.1404007, -0.1878907, 0.1546452
4: -0.0884082, 0.1426494, -0.1195223, 0.1871233, -0.2755313, 0.2621717

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0892142, upper bound: 0.0882117
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0892142, upper bound: 0.0882400
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0200956, 0.0350137, -0.0358490, 0.0487337, -0.0688294, 0.0708628
1: -0.0474771, 0.1145338, -0.0507856, 0.1154432, -0.1629202, 0.1653194
2: -0.0544097, 0.1152232, -0.1024819, 0.1582004, -0.2126101, 0.2177052
3: -0.0536702, 0.1447884, -0.0580275, 0.1404007, -0.1940709, 0.2028158
4: -0.0832517, 0.1436815, -0.1195223, 0.1871233, -0.2703749, 0.2632038

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882742, upper bound: 0.0877213
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883129, upper bound: 0.0877368
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0281048, 0.0310157, -0.0262249, 0.0476892, -0.0757940, 0.0572406
1: -0.0405944, 0.0818225, -0.0560731, 0.1461119, -0.1867063, 0.1378957
2: -0.0773178, 0.1205083, -0.0741902, 0.1451171, -0.2224349, 0.1946985
3: -0.0474758, 0.0965890, -0.0633892, 0.1894719, -0.2369477, 0.1599782
4: -0.0883616, 0.1425957, -0.1138687, 0.1800817, -0.2684433, 0.2564643

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885755, upper bound: 0.0881717
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885870, upper bound: 0.0881785
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0200956, 0.0350137, -0.0262249, 0.0476892, -0.0677848, 0.0612386
1: -0.0474771, 0.1145338, -0.0560731, 0.1461119, -0.1935890, 0.1706070
2: -0.0544097, 0.1152232, -0.0741902, 0.1451171, -0.1995267, 0.1894134
3: -0.0536702, 0.1447884, -0.0633892, 0.1894719, -0.2431421, 0.2081776
4: -0.0832517, 0.1436815, -0.1138687, 0.1800817, -0.2633333, 0.2575501

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879745, upper bound: 0.0879238
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879745, upper bound: 0.0879238
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0326413, 0.0406024, -0.0358490, 0.0487337, -0.0813750, 0.0764514
1: -0.0464028, 0.1029233, -0.0507856, 0.1154432, -0.1618460, 0.1537088
2: -0.0931736, 0.1439410, -0.1024819, 0.1582004, -0.2513740, 0.2464229
3: -0.0537068, 0.1251042, -0.0580275, 0.1404007, -0.1941074, 0.1831317
4: -0.1092366, 0.1700594, -0.1195223, 0.1871233, -0.2963600, 0.2895818

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895709, upper bound: 0.0888255
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895709, upper bound: 0.0888528
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0252827, 0.0445866, -0.0358490, 0.0487337, -0.0740165, 0.0804356
1: -0.0539653, 0.1393512, -0.0507856, 0.1154432, -0.1694084, 0.1901368
2: -0.0712305, 0.1388832, -0.1024819, 0.1582004, -0.2294308, 0.2413651
3: -0.0606718, 0.1808298, -0.0580275, 0.1404007, -0.2010725, 0.2388572
4: -0.1092791, 0.1714957, -0.1195223, 0.1871233, -0.2964023, 0.2910181

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896410, upper bound: 0.0888413
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896430, upper bound: 0.0888528
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0320398, 0.0394607, -0.0262249, 0.0476892, -0.0797290, 0.0656856
1: -0.0457833, 0.1013878, -0.0560731, 0.1461119, -0.1918952, 0.1574610
2: -0.0914667, 0.1419749, -0.0741902, 0.1451171, -0.2365838, 0.2161651
3: -0.0532284, 0.1232585, -0.0633892, 0.1894719, -0.2427003, 0.1866477
4: -0.1076469, 0.1679725, -0.1138687, 0.1800817, -0.2877285, 0.2818412

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887795, upper bound: 0.0887843
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0252827, 0.0445866, -0.0262249, 0.0476892, -0.0729719, 0.0708114
1: -0.0539653, 0.1393512, -0.0560731, 0.1461119, -0.2000772, 0.1954244
2: -0.0712305, 0.1388832, -0.0741902, 0.1451171, -0.2163476, 0.2130734
3: -0.0606718, 0.1808298, -0.0633892, 0.1894719, -0.2501436, 0.2442190
4: -0.1092791, 0.1714957, -0.1138687, 0.1800817, -0.2893607, 0.2853644

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879318, upper bound: 0.0879721
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880213
time: 0.34 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.81 seconds
IS_A1_B1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0885416, upper bound: 0.0886151
IS_A1_B1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0884416, upper bound: 0.0883745
IS_A1_B1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0883070, upper bound: 0.0885059
IS_A1_B1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0882591, upper bound: 0.0882591
IS_A1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0884127, upper bound: 0.0884950
IS_A1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0884127, upper bound: 0.0886361
IS_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0885535, upper bound: 0.0884950
IS_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0885535, upper bound: 0.0886361
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0884127, upper bound: 0.0884127
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0884127, upper bound: 0.0884127
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0885535
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0885535
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0884192
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0884192
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0885564
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0885564
IS_A1_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0886327, upper bound: 0.0889554
IS_A1_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0886327, upper bound: 0.0889554
IS_A1_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0886327, upper bound: 0.0889640
IS_A1_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0886327, upper bound: 0.0889640
IS_A1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0885475, upper bound: 0.0888740
IS_A1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0885475, upper bound: 0.0888740
IS_A1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0885475, upper bound: 0.0888740
IS_A1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0885475, upper bound: 0.0889614
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0886295, upper bound: 0.0888141
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0886295, upper bound: 0.0888830
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0886295, upper bound: 0.0888737
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0886295, upper bound: 0.0889766
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0886415, upper bound: 0.0888164
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0885475, upper bound: 0.0888835
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0886415, upper bound: 0.0888810
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0886415, upper bound: 0.0889797
IS_A2_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0889554, upper bound: 0.0886327
IS_A2_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0889554, upper bound: 0.0887379
IS_A2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0889554, upper bound: 0.0886327
IS_A2_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0889554, upper bound: 0.0887379
IS_A2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0888141, upper bound: 0.0885475
IS_A2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0888141, upper bound: 0.0887906
IS_A2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0888141, upper bound: 0.0885596
IS_A2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0888141, upper bound: 0.0887906
IS_A2_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0888141, upper bound: 0.0886295
IS_A2_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0888141, upper bound: 0.0888726
IS_A2_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0888737, upper bound: 0.0886295
IS_A2_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0888737, upper bound: 0.0886295
IS_A2_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0888427, upper bound: 0.0885920
IS_A2_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0888427, upper bound: 0.0887386
IS_A2_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0889193, upper bound: 0.0885920
IS_A2_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0889193, upper bound: 0.0885920
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0892142, upper bound: 0.0882117
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0892142, upper bound: 0.0882400
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0882742, upper bound: 0.0877213
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0883129, upper bound: 0.0877368
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0885755, upper bound: 0.0881717
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0885870, upper bound: 0.0881785
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0879745, upper bound: 0.0879238
IS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0879745, upper bound: 0.0879238
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0895709, upper bound: 0.0888255
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0895709, upper bound: 0.0888528
IS_A2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0896410, upper bound: 0.0888413
IS_A2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0896430, upper bound: 0.0888528
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0887795, upper bound: 0.0887843
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0879318, upper bound: 0.0879721
IS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.81
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880213

## BFS IS instance: IS_A1_B1_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0278064, 0.0288961, -0.0183061, 0.0140802, -0.0418866, 0.0472022
1: -0.0350639, 0.0669351, -0.0240727, 0.0309017, -0.0659656, 0.0910077
2: -0.0795793, 0.1169136, -0.0450367, 0.0758257, -0.1554050, 0.1619503
3: -0.0413456, 0.0759878, -0.0263939, 0.0274213, -0.0687668, 0.1023817
4: -0.0828490, 0.1328639, -0.0390864, 0.0815583, -0.1644074, 0.1719503

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884416, upper bound: 0.0883745
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884416, upper bound: 0.0883745
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0278064, 0.0288961, -0.0182536, 0.0148930, -0.0426994, 0.0471497
1: -0.0350639, 0.0669351, -0.0242180, 0.0328669, -0.0679308, 0.0911531
2: -0.0795793, 0.1169136, -0.0444218, 0.0763963, -0.1559756, 0.1613355
3: -0.0413456, 0.0759878, -0.0269134, 0.0300028, -0.0713484, 0.1029013
4: -0.0828490, 0.1328639, -0.0399020, 0.0830934, -0.1659424, 0.1727659

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884416, upper bound: 0.0883745
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884416, upper bound: 0.0883745
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0270086, 0.0252807, -0.0235625, 0.0223077, -0.0493163, 0.0488432
1: -0.0336199, 0.0582085, -0.0298257, 0.0489782, -0.0825981, 0.0880342
2: -0.0777968, 0.1109209, -0.0665481, 0.0995724, -0.1773691, 0.1774690
3: -0.0399900, 0.0625573, -0.0366166, 0.0498813, -0.0898713, 0.0991738
4: -0.0778713, 0.1258984, -0.0645988, 0.1130393, -0.1909106, 0.1904972

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882591, upper bound: 0.0882591
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882591, upper bound: 0.0882591
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0270086, 0.0252807, -0.0268614, 0.0244703, -0.0514790, 0.0521421
1: -0.0336199, 0.0582085, -0.0328907, 0.0524811, -0.0861010, 0.0910992
2: -0.0777968, 0.1109209, -0.0769134, 0.1082844, -0.1860811, 0.1878343
3: -0.0399900, 0.0625573, -0.0390857, 0.0535550, -0.0935450, 0.1016430
4: -0.0778713, 0.1258984, -0.0741939, 0.1224238, -0.2002951, 0.2000922

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882591, upper bound: 0.0882591
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882591, upper bound: 0.0882591
time: 0.31 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0189038, 0.0158268, -0.0237667, 0.0206726, -0.0395764, 0.0395935
1: -0.0248756, 0.0358514, -0.0301767, 0.0481623, -0.0730379, 0.0660281
2: -0.0468024, 0.0796161, -0.0622640, 0.0955307, -0.1423331, 0.1418801
3: -0.0280160, 0.0339445, -0.0332325, 0.0495364, -0.0775524, 0.0671770
4: -0.0429375, 0.0872777, -0.0597730, 0.1052151, -0.1481525, 0.1470508

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885416, upper bound: 0.0885878
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 36

Time for candidate selection: 2.11 seconds

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 49
type: A, layer: 5, pos: 49
type: B, layer: 5, pos: 33
type: A, layer: 5, pos: 33
type: A, layer: 5, pos: 21
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 21
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 1
type: B, layer: 5, pos: 23
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 34
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 16
type: A, layer: 5, pos: 42
type: A, layer: 5, pos: 23
type: B, layer: 5, pos: 42
type: A, layer: 5, pos: 26
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 14
type: B, layer: 5, pos: 16

Time for candidate selection: 5.47 seconds

### Candidate
type: B, layer: 5, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 49

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0872418, upper bound: 0.0865724
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882219, upper bound: 0.0881715
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0274795, 0.0253483, -0.0237667, 0.0206726, -0.0481521, 0.0491150
1: -0.0339634, 0.0554254, -0.0301767, 0.0481623, -0.0821258, 0.0856021
2: -0.0789161, 0.1109374, -0.0622640, 0.0955307, -0.1744468, 0.1732014
3: -0.0402230, 0.0574996, -0.0332325, 0.0495364, -0.0897594, 0.0907322
4: -0.0770320, 0.1258029, -0.0597730, 0.1052151, -0.1822471, 0.1855760

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885416, upper bound: 0.0886974
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36

Time for candidate selection: 2.10 seconds

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 33
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 23
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 23
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 16
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 26
type: B, layer: 5, pos: 49
type: A, layer: 5, pos: 1
type: B, layer: 5, pos: 1
type: B, layer: 5, pos: 16
type: B, layer: 5, pos: 14
type: B, layer: 5, pos: 42

Time for candidate selection: 5.47 seconds

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0872418, upper bound: 0.0865724
time: 0.29 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882219, upper bound: 0.0882456
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0189038, 0.0158268, -0.0322836, 0.0334207, -0.0523245, 0.0481104
1: -0.0248756, 0.0358514, -0.0396060, 0.0710703, -0.0959459, 0.0754574
2: -0.0468024, 0.0796161, -0.0933639, 0.1287424, -0.1755448, 0.1729800
3: -0.0280160, 0.0339445, -0.0451630, 0.0779622, -0.1059782, 0.0791075
4: -0.0429375, 0.0872777, -0.0942796, 0.1455511, -0.1884885, 0.1815574

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881659, upper bound: 0.0884308
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 21

Time for candidate selection: 2.09 seconds

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 33
type: A, layer: 5, pos: 33
type: A, layer: 5, pos: 26
type: B, layer: 5, pos: 26
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 1
type: B, layer: 5, pos: 23
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 16
type: B, layer: 5, pos: 1
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 34
type: A, layer: 5, pos: 23
type: A, layer: 5, pos: 49
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 28
type: A, layer: 5, pos: 42

Time for candidate selection: 5.42 seconds

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0868762, upper bound: 0.0863725
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878513, upper bound: 0.0879620
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0274795, 0.0253483, -0.0322836, 0.0334207, -0.0609002, 0.0576319
1: -0.0339634, 0.0554254, -0.0396060, 0.0710703, -0.1050337, 0.0950314
2: -0.0789161, 0.1109374, -0.0933639, 0.1287424, -0.2076585, 0.2043013
3: -0.0402230, 0.0574996, -0.0451630, 0.0779622, -0.1181852, 0.1026626
4: -0.0770320, 0.1258029, -0.0942796, 0.1455511, -0.2225831, 0.2200826

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881659, upper bound: 0.0885881
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11

Time for candidate selection: 2.23 seconds

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 33
type: A, layer: 5, pos: 33
type: A, layer: 5, pos: 26
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 21
type: B, layer: 5, pos: 26
type: B, layer: 5, pos: 23
type: A, layer: 5, pos: 15
type: A, layer: 5, pos: 23
type: B, layer: 5, pos: 15
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 28
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 16
type: A, layer: 5, pos: 1
type: B, layer: 5, pos: 16
type: B, layer: 5, pos: 1

Time for candidate selection: 5.84 seconds

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 26

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0868762, upper bound: 0.0863725
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878513, upper bound: 0.0881033
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0189038, 0.0158268, -0.0395935, 0.0395764
1: -0.0301767, 0.0481623, -0.0248756, 0.0358514, -0.0660281, 0.0730379
2: -0.0622640, 0.0955307, -0.0468024, 0.0796161, -0.1418801, 0.1423331
3: -0.0332325, 0.0495364, -0.0280160, 0.0339445, -0.0671770, 0.0775524
4: -0.0597730, 0.1052151, -0.0429375, 0.0872777, -0.1470508, 0.1481525

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885878, upper bound: 0.0885416
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 36

Time for candidate selection: 2.32 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 49
type: B, layer: 5, pos: 49
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 33
type: B, layer: 5, pos: 21
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 21
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 23
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 34
type: A, layer: 5, pos: 1
type: B, layer: 5, pos: 16
type: B, layer: 5, pos: 42
type: B, layer: 5, pos: 23
type: A, layer: 5, pos: 42
type: B, layer: 5, pos: 26
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 14
type: A, layer: 5, pos: 16

Time for candidate selection: 5.95 seconds

### Candidate
type: A, layer: 5, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0864476, upper bound: 0.0872418
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881715, upper bound: 0.0882219
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0274795, 0.0253483, -0.0491150, 0.0481521
1: -0.0301767, 0.0481623, -0.0339634, 0.0554254, -0.0856021, 0.0821258
2: -0.0622640, 0.0955307, -0.0789161, 0.1109374, -0.1732014, 0.1744468
3: -0.0332325, 0.0495364, -0.0402230, 0.0574996, -0.0907322, 0.0897594
4: -0.0597730, 0.1052151, -0.0770320, 0.1258029, -0.1855760, 0.1822471

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885257, upper bound: 0.0885416
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36

Time for candidate selection: 2.28 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 33
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 23
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 23
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 16
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 26
type: A, layer: 5, pos: 49
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 1
type: A, layer: 5, pos: 16
type: A, layer: 5, pos: 14
type: A, layer: 5, pos: 42

Time for candidate selection: 5.96 seconds

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0865724, upper bound: 0.0872418
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881715, upper bound: 0.0882219
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0322836, 0.0334207, -0.0189038, 0.0158268, -0.0481104, 0.0523245
1: -0.0396060, 0.0710703, -0.0248756, 0.0358514, -0.0754574, 0.0959459
2: -0.0933639, 0.1287424, -0.0468024, 0.0796161, -0.1729800, 0.1755448
3: -0.0451630, 0.0779622, -0.0280160, 0.0339445, -0.0791075, 0.1059782
4: -0.0942796, 0.1455511, -0.0429375, 0.0872777, -0.1815574, 0.1884885

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884308, upper bound: 0.0883067
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 21

Time for candidate selection: 2.30 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 33
type: B, layer: 5, pos: 26
type: A, layer: 5, pos: 26
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 23
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 1
type: A, layer: 5, pos: 16
type: B, layer: 5, pos: 34
type: B, layer: 5, pos: 23
type: B, layer: 5, pos: 49
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 28
type: B, layer: 5, pos: 42

Time for candidate selection: 5.70 seconds

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0863725, upper bound: 0.0870174
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879620, upper bound: 0.0879925
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0322836, 0.0334207, -0.0274795, 0.0253483, -0.0576319, 0.0609002
1: -0.0396060, 0.0710703, -0.0339634, 0.0554254, -0.0950314, 0.1050337
2: -0.0933639, 0.1287424, -0.0789161, 0.1109374, -0.2043013, 0.2076585
3: -0.0451630, 0.0779622, -0.0402230, 0.0574996, -0.1026626, 0.1181852
4: -0.0942796, 0.1455511, -0.0770320, 0.1258029, -0.2200826, 0.2225831

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884308, upper bound: 0.0883067
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11

Time for candidate selection: 2.14 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 33
type: B, layer: 5, pos: 26
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 21
type: A, layer: 5, pos: 26
type: A, layer: 5, pos: 23
type: B, layer: 5, pos: 15
type: B, layer: 5, pos: 23
type: A, layer: 5, pos: 15
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 28
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 16
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 16
type: A, layer: 5, pos: 1

Time for candidate selection: 5.56 seconds

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0863725, upper bound: 0.0870174
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879620, upper bound: 0.0879938
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0237667, 0.0206726, -0.0444393, 0.0444393
1: -0.0301767, 0.0481623, -0.0301767, 0.0481623, -0.0783390, 0.0783390
2: -0.0622640, 0.0955307, -0.0622640, 0.0955307, -0.1577947, 0.1577947
3: -0.0332325, 0.0495364, -0.0332325, 0.0495364, -0.0827689, 0.0827689
4: -0.0597730, 0.1052151, -0.0597730, 0.1052151, -0.1649881, 0.1649881

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11

Time for candidate selection: 1.83 seconds

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 49
type: B, layer: 5, pos: 49
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 33
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 23
type: B, layer: 5, pos: 23
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 14
type: B, layer: 5, pos: 14
type: A, layer: 5, pos: 1
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 16
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 42
type: B, layer: 5, pos: 42

Time for candidate selection: 5.34 seconds

### Candidate
type: A, layer: 5, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0874474, upper bound: 0.0869530
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881715, upper bound: 0.0882792
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0322791, 0.0333924, -0.0571591, 0.0529517
1: -0.0301767, 0.0481623, -0.0395984, 0.0710182, -0.1011949, 0.0877608
2: -0.0622640, 0.0955307, -0.0933608, 0.1287068, -0.1909709, 0.1888915
3: -0.0332325, 0.0495364, -0.0451363, 0.0779050, -0.1111376, 0.0946727
4: -0.0597730, 0.1052151, -0.0942774, 0.1454905, -0.2052636, 0.1994925

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 11

Time for candidate selection: 1.83 seconds

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 33
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 15
type: A, layer: 5, pos: 23
type: B, layer: 5, pos: 23
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 1
type: A, layer: 5, pos: 49
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 16
type: B, layer: 5, pos: 26
type: A, layer: 5, pos: 14
type: A, layer: 5, pos: 42

Time for candidate selection: 5.56 seconds

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0866728, upper bound: 0.0873629
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881715, upper bound: 0.0882792
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0322836, 0.0334207, -0.0237667, 0.0206726, -0.0529562, 0.0571874
1: -0.0396060, 0.0710703, -0.0301767, 0.0481623, -0.0877684, 0.1012470
2: -0.0933639, 0.1287424, -0.0622640, 0.0955307, -0.1888946, 0.1910064
3: -0.0451630, 0.0779622, -0.0332325, 0.0495364, -0.0946994, 0.1111947
4: -0.0942796, 0.1455511, -0.0597730, 0.1052151, -0.1994947, 0.2053241

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 11

Time for candidate selection: 2.00 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 33
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 15
type: B, layer: 5, pos: 23
type: A, layer: 5, pos: 23
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 16
type: B, layer: 5, pos: 1
type: B, layer: 5, pos: 49
type: A, layer: 5, pos: 1
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 26
type: B, layer: 5, pos: 14
type: B, layer: 5, pos: 42

Time for candidate selection: 5.72 seconds

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0872085, upper bound: 0.0864774
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879620, upper bound: 0.0880275
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0322836, 0.0334207, -0.0322791, 0.0333924, -0.0656760, 0.0656997
1: -0.0396060, 0.0710703, -0.0395984, 0.0710182, -0.1106243, 0.1106687
2: -0.0933639, 0.1287424, -0.0933608, 0.1287068, -0.2220708, 0.2221032
3: -0.0451630, 0.0779622, -0.0451363, 0.0779050, -0.1230680, 0.1230985
4: -0.0942796, 0.1455511, -0.0942774, 0.1454905, -0.2397702, 0.2398285

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11

Time for candidate selection: 2.01 seconds

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 33
type: A, layer: 5, pos: 26
type: B, layer: 5, pos: 26
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 21
type: B, layer: 5, pos: 23
type: A, layer: 5, pos: 23
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 34
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 16
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 1
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 28

Time for candidate selection: 5.70 seconds

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 26

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0872085, upper bound: 0.0864774
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879620, upper bound: 0.0880275
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0189038, 0.0158268, -0.0220569, 0.0217685, -0.0406723, 0.0378837
1: -0.0248756, 0.0358514, -0.0301752, 0.0556444, -0.0805200, 0.0660265
2: -0.0468024, 0.0796161, -0.0553310, 0.0897710, -0.1365734, 0.1349471
3: -0.0280160, 0.0339445, -0.0349494, 0.0610096, -0.0890256, 0.0688938
4: -0.0429375, 0.0872777, -0.0578914, 0.1030896, -0.1460271, 0.1451692

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883886, upper bound: 0.0888497
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883838, upper bound: 0.0888444
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0189038, 0.0158268, -0.0266707, 0.0291590, -0.0480628, 0.0424975
1: -0.0248756, 0.0358514, -0.0377345, 0.0736009, -0.0984765, 0.0735859
2: -0.0468024, 0.0796161, -0.0740767, 0.1152196, -0.1620220, 0.1536928
3: -0.0280160, 0.0339445, -0.0447126, 0.0836212, -0.1116372, 0.0786570
4: -0.0429375, 0.0872777, -0.0818900, 0.1354395, -0.1783769, 0.1691677

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883389, upper bound: 0.0888497
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883838, upper bound: 0.0888444
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0189038, 0.0158268, -0.0266258, 0.0277274, -0.0466312, 0.0424526
1: -0.0248756, 0.0358514, -0.0362590, 0.0729906, -0.0978662, 0.0721103
2: -0.0468024, 0.0796161, -0.0705321, 0.1095358, -0.1563383, 0.1501482
3: -0.0280160, 0.0339445, -0.0412077, 0.0849826, -0.1129986, 0.0751522
4: -0.0429375, 0.0872777, -0.0779760, 0.1265911, -0.1695286, 0.1652537

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883389, upper bound: 0.0888586
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 35

Time for candidate selection: 2.28 seconds

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881132, upper bound: 0.0887307
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885697, upper bound: 0.0888956
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0875159, upper bound: 0.0878515
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886291, upper bound: 0.0889620
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 49
type: B, layer: 5, pos: 49
type: B, layer: 5, pos: 26
type: A, layer: 5, pos: 26
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 24
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 15
type: B, layer: 5, pos: 2
type: A, layer: 5, pos: 1
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 23
type: B, layer: 5, pos: 1
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 23
type: A, layer: 5, pos: 16
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 28
type: B, layer: 5, pos: 14
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 48
type: A, layer: 5, pos: 42

Time for candidate selection: 8.44 seconds

### Candidate
type: A, layer: 5, pos: 49

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 49

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 26

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 26

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0864535, upper bound: 0.0874433
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880987, upper bound: 0.0885966
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0189038, 0.0158268, -0.0325525, 0.0409533, -0.0598571, 0.0483793
1: -0.0248756, 0.0358514, -0.0455905, 0.0964378, -0.1213134, 0.0814419
2: -0.0468024, 0.0796161, -0.0924400, 0.1412830, -0.1880854, 0.1720561
3: -0.0280160, 0.0339445, -0.0517285, 0.1139276, -0.1419436, 0.0856730
4: -0.0429375, 0.0872777, -0.1049226, 0.1652986, -0.2082361, 0.1922003

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883886, upper bound: 0.0888586
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 35

Time for candidate selection: 2.25 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0875159, upper bound: 0.0878515
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881132, upper bound: 0.0887309
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885697, upper bound: 0.0888956
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 35

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886291, upper bound: 0.0889621
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 33
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 26
type: A, layer: 5, pos: 26
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 15
type: B, layer: 5, pos: 24
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 2
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 1
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 49
type: A, layer: 5, pos: 23
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 16
type: B, layer: 5, pos: 23
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 14
type: B, layer: 5, pos: 48
type: A, layer: 5, pos: 42

Time for candidate selection: 8.41 seconds

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 26

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 26

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0863546, upper bound: 0.0871687
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880987, upper bound: 0.0885966
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0274795, 0.0253483, -0.0220569, 0.0217685, -0.0492480, 0.0474051
1: -0.0339634, 0.0554254, -0.0301752, 0.0556444, -0.0896078, 0.0856005
2: -0.0789161, 0.1109374, -0.0553310, 0.0897710, -0.1686871, 0.1662684
3: -0.0402230, 0.0574996, -0.0349494, 0.0610096, -0.1012327, 0.0924490
4: -0.0770320, 0.1258029, -0.0578914, 0.1030896, -0.1801216, 0.1836943

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883009, upper bound: 0.0888260
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882963, upper bound: 0.0887677
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0274795, 0.0253483, -0.0266707, 0.0291590, -0.0566385, 0.0520190
1: -0.0339634, 0.0554254, -0.0377345, 0.0736009, -0.1075643, 0.0931598
2: -0.0789161, 0.1109374, -0.0740767, 0.1152196, -0.1941357, 0.1850141
3: -0.0402230, 0.0574996, -0.0447126, 0.0836212, -0.1238442, 0.1022122
4: -0.0770320, 0.1258029, -0.0818900, 0.1354395, -0.2124715, 0.2076929

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883009, upper bound: 0.0888261
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882963, upper bound: 0.0888959
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0274795, 0.0253483, -0.0266258, 0.0277274, -0.0552069, 0.0519741
1: -0.0339634, 0.0554254, -0.0362590, 0.0729906, -0.1069540, 0.0916843
2: -0.0789161, 0.1109374, -0.0705321, 0.1095358, -0.1884519, 0.1814695
3: -0.0402230, 0.0574996, -0.0412077, 0.0849826, -0.1252056, 0.0987073
4: -0.0770320, 0.1258029, -0.0779760, 0.1265911, -0.2036232, 0.2037789

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883131, upper bound: 0.0888717
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 36
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 35

Time for candidate selection: 2.40 seconds

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885013, upper bound: 0.0888019
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877779, upper bound: 0.0885081
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885536, upper bound: 0.0889180
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 26
type: B, layer: 5, pos: 26
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 24
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 16
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 1
type: A, layer: 5, pos: 23
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 1
type: B, layer: 5, pos: 2
type: B, layer: 5, pos: 23
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 16
type: B, layer: 5, pos: 28
type: B, layer: 5, pos: 49
type: B, layer: 5, pos: 14
type: B, layer: 5, pos: 48

Time for candidate selection: 8.43 seconds

### Candidate
type: A, layer: 5, pos: 26

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 26

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0862971, upper bound: 0.0873978
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B1_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880469, upper bound: 0.0885119
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0274795, 0.0253483, -0.0325525, 0.0409533, -0.0684328, 0.0579007
1: -0.0339634, 0.0554254, -0.0455905, 0.0964378, -0.1304012, 0.1010159
2: -0.0789161, 0.1109374, -0.0924400, 0.1412830, -0.2201991, 0.2033774
3: -0.0402230, 0.0574996, -0.0517285, 0.1139276, -0.1541506, 0.1092281
4: -0.0770320, 0.1258029, -0.1049226, 0.1652986, -0.2423307, 0.2307255

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883131, upper bound: 0.0889619
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 35

Time for candidate selection: 2.43 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885013, upper bound: 0.0888971
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0877779, upper bound: 0.0887695
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 35

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885536, upper bound: 0.0889975
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 33
type: B, layer: 5, pos: 26
type: A, layer: 5, pos: 26
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 24
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 16
type: A, layer: 5, pos: 1
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 23
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 2
type: B, layer: 5, pos: 1
type: B, layer: 5, pos: 23
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 28
type: B, layer: 5, pos: 14
type: B, layer: 5, pos: 48

Time for candidate selection: 8.47 seconds

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 26

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 26

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0862971, upper bound: 0.0873978
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_B2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880469, upper bound: 0.0886343
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0220569, 0.0217685, -0.0455352, 0.0427295
1: -0.0301767, 0.0481623, -0.0301752, 0.0556444, -0.0858211, 0.0783375
2: -0.0622640, 0.0955307, -0.0553310, 0.0897710, -0.1520350, 0.1508617
3: -0.0332325, 0.0495364, -0.0349494, 0.0610096, -0.0942422, 0.0844857
4: -0.0597730, 0.1052151, -0.0578914, 0.1030896, -0.1628626, 0.1631065

Time for backsubstitution: 1.91 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0500000, mid=0.0500000, abs_max=0.10251016169786453
rel_dist={0: [-0.0899835175585181, 0.08998351755851813]}

## Binary search (step 2) starts
Candidate diff: 0.0250000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0897251, upper bound: 0.0897771
time: 0.31 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0899572, upper bound: 0.0899572
time: 0.38 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.84 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.84
Output dim: 0, lower bound: -0.0897251, upper bound: 0.0897771
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.84
Output dim: 0, lower bound: -0.0899572, upper bound: 0.0899572

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0355980, 0.0491149, -0.0348397, 0.0414031, -0.0770011, 0.0839546
1: -0.0454062, 0.1038414, -0.0484806, 0.0959572, -0.1413634, 0.1523219
2: -0.1046092, 0.1549752, -0.0997171, 0.1459962, -0.2506055, 0.2546923
3: -0.0520488, 0.1242788, -0.0542691, 0.1110557, -0.1631045, 0.1785479
4: -0.1149560, 0.1775723, -0.1078875, 0.1718937, -0.2868497, 0.2854598

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0897144, upper bound: 0.0897737
time: 0.30 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0897251, upper bound: 0.0897771
time: 0.31 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0383316, 0.0603406, -0.0388105, 0.0622164, -0.1005480, 0.0991511
1: -0.0543237, 0.1360877, -0.0552332, 0.1390822, -0.1934059, 0.1913209
2: -0.1110429, 0.1767848, -0.1125810, 0.1800085, -0.2910514, 0.2893658
3: -0.0620208, 0.1679080, -0.0630197, 0.1716416, -0.2336624, 0.2309277
4: -0.1327446, 0.2087907, -0.1348271, 0.2127281, -0.3454726, 0.3436178

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0897771, upper bound: 0.0897251
time: 0.34 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0897771, upper bound: 0.0897251
time: 0.32 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.25 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -0.0897144, upper bound: 0.0897737
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -0.0897251, upper bound: 0.0897771
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -0.0897771, upper bound: 0.0897251
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.25
Output dim: 0, lower bound: -0.0897771, upper bound: 0.0897251

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0355980, 0.0491149, -0.0276000, 0.0272872, -0.0628853, 0.0767149
1: -0.0454062, 0.1038414, -0.0387172, 0.0671747, -0.1125809, 0.1425585
2: -0.1046092, 0.1549752, -0.0765237, 0.1130567, -0.2176660, 0.2314989
3: -0.0520488, 0.1242788, -0.0446528, 0.0731699, -0.1252187, 0.1689316
4: -0.1149560, 0.1775723, -0.0798540, 0.1326616, -0.2476175, 0.2574263

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896776, upper bound: 0.0897308
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896592, upper bound: 0.0897719
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896592, upper bound: 0.0897737
time: 0.30 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0355980, 0.0491149, -0.0318823, 0.0345591, -0.0701572, 0.0809972
1: -0.0454062, 0.1038414, -0.0443314, 0.0845775, -0.1299837, 0.1481728
2: -0.1046092, 0.1549752, -0.0910832, 0.1329029, -0.2375122, 0.2460583
3: -0.0520488, 0.1242788, -0.0501258, 0.0969920, -0.1490408, 0.1744045
4: -0.1149560, 0.1775723, -0.0985514, 0.1559027, -0.2708587, 0.2761236

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896693, upper bound: 0.0897757
time: 0.30 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896693, upper bound: 0.0897771
time: 0.31 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0383316, 0.0603406, -0.0355980, 0.0491149, -0.0874465, 0.0959386
1: -0.0543237, 0.1360877, -0.0454062, 0.1038414, -0.1581650, 0.1814938
2: -0.1110429, 0.1767848, -0.1046092, 0.1549752, -0.2660180, 0.2813941
3: -0.0620208, 0.1679080, -0.0520488, 0.1242788, -0.1862996, 0.2199568
4: -0.1327446, 0.2087907, -0.1149560, 0.1775723, -0.3103168, 0.3237466

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0897757, upper bound: 0.0896693
time: 0.30 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0897771, upper bound: 0.0897251
time: 0.31 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0383316, 0.0603406, -0.0383316, 0.0603406, -0.0986722, 0.0986722
1: -0.0543237, 0.1360877, -0.0543237, 0.1360877, -0.1904113, 0.1904114
2: -0.1110429, 0.1767848, -0.1110429, 0.1767848, -0.2878277, 0.2878277
3: -0.0620208, 0.1679080, -0.0620208, 0.1679080, -0.2299288, 0.2299288
4: -0.1327446, 0.2087907, -0.1327446, 0.2087907, -0.3415352, 0.3415352

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0897737, upper bound: 0.0898844
time: 0.32 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0897771, upper bound: 0.0899088
time: 0.32 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.93 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -0.0896592, upper bound: 0.0897719
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -0.0896592, upper bound: 0.0897737
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -0.0896693, upper bound: 0.0897757
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -0.0896693, upper bound: 0.0897771
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -0.0897757, upper bound: 0.0896693
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -0.0897771, upper bound: 0.0897251
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -0.0897737, upper bound: 0.0898844
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -0.0897771, upper bound: 0.0899088

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0276000, 0.0272872, -0.0562071, 0.0592198
1: -0.0364605, 0.0725038, -0.0387172, 0.0671747, -0.1036352, 0.1112209
2: -0.0836895, 0.1224094, -0.0765237, 0.1130567, -0.1967463, 0.1989331
3: -0.0432477, 0.0836505, -0.0446528, 0.0731699, -0.1164175, 0.1283033
4: -0.0884080, 0.1397506, -0.0798540, 0.1326616, -0.2210695, 0.2196046

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896206, upper bound: 0.0897253
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896271, upper bound: 0.0896271
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896271, upper bound: 0.0896271
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0341963, 0.0451829, -0.0276000, 0.0272872, -0.0614835, 0.0727828
1: -0.0427969, 0.0966445, -0.0387172, 0.0671747, -0.1099716, 0.1353617
2: -0.1001576, 0.1478489, -0.0765237, 0.1130567, -0.2132143, 0.2243726
3: -0.0493405, 0.1152938, -0.0446528, 0.0731699, -0.1225104, 0.1599466
4: -0.1093097, 0.1684246, -0.0798540, 0.1326616, -0.2419713, 0.2482786

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896206, upper bound: 0.0897308
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896271, upper bound: 0.0896273
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896271, upper bound: 0.0896273
time: 0.31 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0318823, 0.0345591, -0.0634790, 0.0635022
1: -0.0364605, 0.0725038, -0.0443314, 0.0845775, -0.1210380, 0.1168352
2: -0.0836895, 0.1224094, -0.0910832, 0.1329029, -0.2165925, 0.2134926
3: -0.0432477, 0.0836505, -0.0501258, 0.0969920, -0.1402396, 0.1337762
4: -0.0884080, 0.1397506, -0.0985514, 0.1559027, -0.2443107, 0.2383020

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896271, upper bound: 0.0896811
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896271, upper bound: 0.0897719
time: 0.32 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0341963, 0.0451829, -0.0318823, 0.0345591, -0.0687554, 0.0770651
1: -0.0427969, 0.0966445, -0.0443314, 0.0845775, -0.1273744, 0.1409759
2: -0.1001576, 0.1478489, -0.0910832, 0.1329029, -0.2330605, 0.2389321
3: -0.0493405, 0.1152938, -0.0501258, 0.0969920, -0.1463325, 0.1654195
4: -0.1093097, 0.1684246, -0.0985514, 0.1559027, -0.2652124, 0.2669760

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896271, upper bound: 0.0896455
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0896271, upper bound: 0.0897760
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0383316, 0.0603406, -0.0289199, 0.0316199, -0.0699515, 0.0892605
1: -0.0543237, 0.1360877, -0.0364605, 0.0725038, -0.1268275, 0.1725482
2: -0.1110429, 0.1767848, -0.0836895, 0.1224094, -0.2334523, 0.2604744
3: -0.0620208, 0.1679080, -0.0432477, 0.0836505, -0.1456713, 0.2111557
4: -0.1327446, 0.2087907, -0.0884080, 0.1397506, -0.2724952, 0.2971986

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887840, upper bound: 0.0885736
time: 0.31 seconds

## Relational analysis of IS_A2_B1_B1_B2

### Relational analysis result of IS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888768, upper bound: 0.0886822
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0383316, 0.0603406, -0.0341963, 0.0451829, -0.0835145, 0.0945368
1: -0.0543237, 0.1360877, -0.0427969, 0.0966445, -0.1509682, 0.1788846
2: -0.1110429, 0.1767848, -0.1001576, 0.1478489, -0.2588917, 0.2769423
3: -0.0620208, 0.1679080, -0.0493405, 0.1152938, -0.1773146, 0.2172485
4: -0.1327446, 0.2087907, -0.1093097, 0.1684246, -0.3011691, 0.3181004

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0897737, upper bound: 0.0897144
time: 0.32 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0897737, upper bound: 0.0897251
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0303880, 0.0381159, -0.0383316, 0.0603406, -0.0907286, 0.0764475
1: -0.0436567, 0.0986741, -0.0543237, 0.1360877, -0.1797444, 0.1529978
2: -0.0858032, 0.1358688, -0.1110429, 0.1767848, -0.2625880, 0.2469117
3: -0.0513058, 0.1197162, -0.0620208, 0.1679080, -0.2192139, 0.1817370
4: -0.1010720, 0.1610686, -0.1327446, 0.2087907, -0.3098626, 0.2938131

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887805, upper bound: 0.0882075
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885666, upper bound: 0.0881785
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0352700, 0.0520305, -0.0383316, 0.0603406, -0.0956105, 0.0903621
1: -0.0497746, 0.1236877, -0.0543237, 0.1360877, -0.1858622, 0.1780114
2: -0.1017607, 0.1623120, -0.1110429, 0.1767848, -0.2785456, 0.2733549
3: -0.0576897, 0.1530293, -0.0620208, 0.1679080, -0.2255976, 0.2150501
4: -0.1224101, 0.1915598, -0.1327446, 0.2087907, -0.3312007, 0.3243043

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894665, upper bound: 0.0888373
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
time: 0.35 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.55 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0896271, upper bound: 0.0896271
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0896271, upper bound: 0.0896271
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0896271, upper bound: 0.0896273
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0896271, upper bound: 0.0896273
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0896271, upper bound: 0.0896811
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0896271, upper bound: 0.0897719
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0896271, upper bound: 0.0896455
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0896271, upper bound: 0.0897760
IS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0887840, upper bound: 0.0885736
IS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0888768, upper bound: 0.0886822
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0897737, upper bound: 0.0897144
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0897737, upper bound: 0.0897251
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0887805, upper bound: 0.0882075
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0885666, upper bound: 0.0881785
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0894665, upper bound: 0.0888373
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.55
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0281227, 0.0264112, -0.0553311, 0.0597426
1: -0.0364605, 0.0725038, -0.0358182, 0.0631274, -0.0995880, 0.1083220
2: -0.0836895, 0.1224094, -0.0803205, 0.1135095, -0.1971990, 0.2027299
3: -0.0432477, 0.0836505, -0.0419889, 0.0698837, -0.1131313, 0.1256394
4: -0.0884080, 0.1397506, -0.0817580, 0.1295700, -0.2179780, 0.2215086

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894284, upper bound: 0.0895061
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893074, upper bound: 0.0893074
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0296060, 0.0335974, -0.0625173, 0.0612259
1: -0.0364605, 0.0725038, -0.0426784, 0.0909929, -0.1274535, 0.1151821
2: -0.0836895, 0.1224094, -0.0831003, 0.1289739, -0.2126634, 0.2055098
3: -0.0432477, 0.0836505, -0.0500414, 0.1097184, -0.1529661, 0.1336919
4: -0.0884080, 0.1397506, -0.0968272, 0.1530392, -0.2414472, 0.2365778

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894284, upper bound: 0.0895061
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893074, upper bound: 0.0895949
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0341963, 0.0451829, -0.0281227, 0.0264112, -0.0606074, 0.0733056
1: -0.0427969, 0.0966445, -0.0358182, 0.0631274, -0.1059244, 0.1324627
2: -0.1001576, 0.1478489, -0.0803205, 0.1135095, -0.2136671, 0.2281694
3: -0.0493405, 0.1152938, -0.0419889, 0.0698837, -0.1192242, 0.1572827
4: -0.1093097, 0.1684246, -0.0817580, 0.1295700, -0.2388797, 0.2501824

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885636, upper bound: 0.0886047
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886361, upper bound: 0.0885535
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0341963, 0.0451829, -0.0296060, 0.0335974, -0.0677936, 0.0747889
1: -0.0427969, 0.0966445, -0.0426784, 0.0909929, -0.1337899, 0.1393229
2: -0.1001576, 0.1478489, -0.0831003, 0.1289739, -0.2291314, 0.2309492
3: -0.0493405, 0.1152938, -0.0500414, 0.1097184, -0.1590590, 0.1653352
4: -0.1093097, 0.1684246, -0.0968272, 0.1530392, -0.2623489, 0.2652518

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895602, upper bound: 0.0897308
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885636, upper bound: 0.0887494
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885538, upper bound: 0.0885535
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0336742, 0.0405566, -0.0694765, 0.0652941
1: -0.0364605, 0.0725038, -0.0424529, 0.0908265, -0.1272870, 0.1149567
2: -0.0836895, 0.1224094, -0.0992337, 0.1441604, -0.2278500, 0.2216431
3: -0.0432477, 0.0836505, -0.0490791, 0.1076450, -0.1508926, 0.1327296
4: -0.0884080, 0.1397506, -0.1055799, 0.1649233, -0.2533313, 0.2453305

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894284, upper bound: 0.0895602
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886047, upper bound: 0.0886729
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885535, upper bound: 0.0886361
time: 0.31 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0340651, 0.0448900, -0.0738099, 0.0656850
1: -0.0364605, 0.0725038, -0.0483907, 0.1140353, -0.1504958, 0.1208944
2: -0.0836895, 0.1224094, -0.0988940, 0.1544030, -0.2380925, 0.2213034
3: -0.0432477, 0.0836505, -0.0562355, 0.1411494, -0.1843971, 0.1398859
4: -0.0884080, 0.1397506, -0.1176342, 0.1828300, -0.2712380, 0.2573848

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885631, upper bound: 0.0886598
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885535, upper bound: 0.0886361
time: 0.32 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0341963, 0.0451829, -0.0336742, 0.0405566, -0.0747528, 0.0788571
1: -0.0427969, 0.0966445, -0.0424529, 0.0908265, -0.1336234, 0.1390974
2: -0.1001576, 0.1478489, -0.0992337, 0.1441604, -0.2443180, 0.2470826
3: -0.0493405, 0.1152938, -0.0490791, 0.1076450, -0.1569855, 0.1643729
4: -0.1093097, 0.1684246, -0.1055799, 0.1649233, -0.2742330, 0.2740044

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886724, upper bound: 0.0886139
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886358, upper bound: 0.0885564
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0341963, 0.0451829, -0.0340651, 0.0448900, -0.0790863, 0.0792479
1: -0.0427969, 0.0966445, -0.0483907, 0.1140353, -0.1568322, 0.1450352
2: -0.1001576, 0.1478489, -0.0988940, 0.1544030, -0.2545606, 0.2467429
3: -0.0493405, 0.1152938, -0.0562355, 0.1411494, -0.1904899, 0.1715293
4: -0.1093097, 0.1684246, -0.1176342, 0.1828300, -0.2921397, 0.2860586

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886724, upper bound: 0.0887621
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886358, upper bound: 0.0888293
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0350352, 0.0461892, -0.0189038, 0.0158268, -0.0508620, 0.0650930
1: -0.0492939, 0.1105508, -0.0248756, 0.0358514, -0.0851453, 0.1354264
2: -0.0994778, 0.1517577, -0.0468024, 0.0796161, -0.1790938, 0.1985601
3: -0.0552683, 0.1334484, -0.0280160, 0.0339445, -0.0892128, 0.1614643
4: -0.1133727, 0.1787415, -0.0429375, 0.0872777, -0.2006505, 0.2216789

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886777, upper bound: 0.0882960
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887493, upper bound: 0.0885736
time: 0.30 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887493, upper bound: 0.0885736
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0338814, 0.0446531, -0.0274795, 0.0253483, -0.0592297, 0.0721326
1: -0.0469683, 0.1047603, -0.0339634, 0.0554254, -0.1023936, 0.1387237
2: -0.0967741, 0.1502057, -0.0789161, 0.1109374, -0.2077115, 0.2291218
3: -0.0544668, 0.1253278, -0.0402230, 0.0574996, -0.1119664, 0.1655508
4: -0.1112777, 0.1769101, -0.0770320, 0.1258029, -0.2370806, 0.2539421

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888167, upper bound: 0.0884397
time: 0.32 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888262, upper bound: 0.0886822
time: 0.33 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888262, upper bound: 0.0886822
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0303880, 0.0381159, -0.0341963, 0.0451829, -0.0755709, 0.0723121
1: -0.0436567, 0.0986741, -0.0427969, 0.0966445, -0.1403012, 0.1414710
2: -0.0858032, 0.1358688, -0.1001576, 0.1478489, -0.2336521, 0.2360264
3: -0.0513058, 0.1197162, -0.0493405, 0.1152938, -0.1665996, 0.1690567
4: -0.1010720, 0.1610686, -0.1093097, 0.1684246, -0.2694965, 0.2703783

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887493, upper bound: 0.0887338
time: 0.31 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888261, upper bound: 0.0888025
time: 0.31 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0352700, 0.0520305, -0.0341963, 0.0451829, -0.0804528, 0.0862267
1: -0.0497746, 0.1236877, -0.0427969, 0.0966445, -0.1464191, 0.1664846
2: -0.1017607, 0.1623120, -0.1001576, 0.1478489, -0.2496097, 0.2624695
3: -0.0576897, 0.1530293, -0.0493405, 0.1152938, -0.1729835, 0.2023698
4: -0.1224101, 0.1915598, -0.1093097, 0.1684246, -0.2908346, 0.3008695

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887493, upper bound: 0.0885790
time: 0.34 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0888261, upper bound: 0.0886825
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0303880, 0.0381159, -0.0358490, 0.0487337, -0.0791218, 0.0739649
1: -0.0436567, 0.0986741, -0.0507856, 0.1154432, -0.1590999, 0.1494596
2: -0.0858032, 0.1358688, -0.1024819, 0.1582004, -0.2440036, 0.2383507
3: -0.0513058, 0.1197162, -0.0580275, 0.1404007, -0.1917065, 0.1777436
4: -0.1010720, 0.1610686, -0.1195223, 0.1871233, -0.2881952, 0.2805909

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885666, upper bound: 0.0881785
time: 0.32 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885666, upper bound: 0.0881785
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0227962, 0.0284473, -0.0262249, 0.0476892, -0.0704854, 0.0546722
1: -0.0338879, 0.0812566, -0.0560731, 0.1461119, -0.1799999, 0.1373297
2: -0.0625081, 0.1091304, -0.0741902, 0.1451171, -0.2076252, 0.1833206
3: -0.0422262, 0.1002784, -0.0633892, 0.1894719, -0.2316981, 0.1636676
4: -0.0791439, 0.1289570, -0.1138687, 0.1800817, -0.2592255, 0.2428257

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885666, upper bound: 0.0881785
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885666, upper bound: 0.0881785
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0352700, 0.0520305, -0.0358490, 0.0487337, -0.0840037, 0.0878795
1: -0.0497746, 0.1236877, -0.0507856, 0.1154432, -0.1652177, 0.1744732
2: -0.1017607, 0.1623120, -0.1024819, 0.1582004, -0.2599612, 0.2647940
3: -0.0576897, 0.1530293, -0.0580275, 0.1404007, -0.1980903, 0.2110567
4: -0.1224101, 0.1915598, -0.1195223, 0.1871233, -0.3095332, 0.3110821

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0283087, 0.0370338, -0.0262249, 0.0476892, -0.0759979, 0.0632586
1: -0.0406796, 0.0998553, -0.0560731, 0.1461119, -0.1867915, 0.1559284
2: -0.0804190, 0.1321447, -0.0741902, 0.1451171, -0.2255361, 0.2063349
3: -0.0490869, 0.1239246, -0.0633892, 0.1894719, -0.2385588, 0.1873138
4: -0.0994766, 0.1560164, -0.1138687, 0.1800817, -0.2795583, 0.2698850

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
time: 0.36 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.63 seconds
IS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0894284, upper bound: 0.0895061
IS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0893074, upper bound: 0.0893074
IS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0894284, upper bound: 0.0895061
IS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0893074, upper bound: 0.0895949
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0885636, upper bound: 0.0886047
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0886361, upper bound: 0.0885535
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0885636, upper bound: 0.0887494
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0885538, upper bound: 0.0885535
IS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0886047, upper bound: 0.0886729
IS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0885535, upper bound: 0.0886361
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0885631, upper bound: 0.0886598
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0885535, upper bound: 0.0886361
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0886724, upper bound: 0.0886139
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0886358, upper bound: 0.0885564
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0886724, upper bound: 0.0887621
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0886358, upper bound: 0.0888293
IS_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0887493, upper bound: 0.0885736
IS_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0887493, upper bound: 0.0885736
IS_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0888262, upper bound: 0.0886822
IS_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0888262, upper bound: 0.0886822
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0887493, upper bound: 0.0887338
IS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0888261, upper bound: 0.0888025
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0887493, upper bound: 0.0885790
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0888261, upper bound: 0.0886825
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0885666, upper bound: 0.0881785
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0885666, upper bound: 0.0881785
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0885666, upper bound: 0.0881785
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0885666, upper bound: 0.0881785
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871

## BFS IS instance: IS_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0263475, 0.0250260, -0.0539459, 0.0579674
1: -0.0364605, 0.0725038, -0.0333350, 0.0601690, -0.0966296, 0.1058388
2: -0.0836895, 0.1224094, -0.0742820, 0.1107289, -0.1944184, 0.1966915
3: -0.0432477, 0.0836505, -0.0402154, 0.0662137, -0.1094614, 0.1238659
4: -0.0884080, 0.1397506, -0.0753473, 0.1262361, -0.2146440, 0.2150979

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893074, upper bound: 0.0893074
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893074, upper bound: 0.0893074
time: 0.43 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0274014, 0.0255119, -0.0544318, 0.0590213
1: -0.0364605, 0.0725038, -0.0342971, 0.0605227, -0.0969833, 0.1068008
2: -0.0836895, 0.1224094, -0.0780043, 0.1108751, -0.1945646, 0.2004137
3: -0.0432477, 0.0836505, -0.0405568, 0.0666638, -0.1099115, 0.1242073
4: -0.0884080, 0.1397506, -0.0788346, 0.1258900, -0.2142980, 0.2185852

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893074, upper bound: 0.0893074
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893074, upper bound: 0.0893074
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0234641, 0.0292775, -0.0581974, 0.0550840
1: -0.0364605, 0.0725038, -0.0357470, 0.0820802, -0.1185407, 0.1082508
2: -0.0836895, 0.1224094, -0.0645553, 0.1123769, -0.1960665, 0.1869647
3: -0.0432477, 0.0836505, -0.0443082, 0.1001011, -0.1433488, 0.1279587
4: -0.0884080, 0.1397506, -0.0799981, 0.1339988, -0.2224067, 0.2197487

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894212, upper bound: 0.0895949
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894212, upper bound: 0.0895949
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0289199, 0.0316199, -0.0287184, 0.0323664, -0.0612863, 0.0603383
1: -0.0364605, 0.0725038, -0.0403162, 0.0872710, -0.1237316, 0.1128200
2: -0.0836895, 0.1224094, -0.0803196, 0.1253408, -0.2090303, 0.2027290
3: -0.0432477, 0.0836505, -0.0482766, 0.1049064, -0.1481541, 0.1319271
4: -0.0884080, 0.1397506, -0.0931064, 0.1482157, -0.2366237, 0.2328570

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A1_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894212, upper bound: 0.0895949
time: 0.39 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894212, upper bound: 0.0895949
time: 0.46 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0245406, 0.0217418, -0.0455085, 0.0452132
1: -0.0301767, 0.0481623, -0.0316688, 0.0498504, -0.0800271, 0.0798312
2: -0.0622640, 0.0955307, -0.0664426, 0.0984977, -0.1607617, 0.1619733
3: -0.0332325, 0.0495364, -0.0358780, 0.0518489, -0.0850815, 0.0854144
4: -0.0597730, 0.1052151, -0.0645340, 0.1104091, -0.1701821, 0.1697491

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885286, upper bound: 0.0883836
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0884127
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0884127
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0322836, 0.0334207, -0.0256863, 0.0222448, -0.0545283, 0.0591069
1: -0.0396060, 0.0710703, -0.0319307, 0.0488508, -0.0884568, 0.1030009
2: -0.0933639, 0.1287424, -0.0724024, 0.1027126, -0.1960765, 0.2011448
3: -0.0451630, 0.0779622, -0.0374903, 0.0488638, -0.0940268, 0.1154525
4: -0.0942796, 0.1455511, -0.0686481, 0.1157893, -0.2100689, 0.2141992

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885881, upper bound: 0.0883061
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884127, upper bound: 0.0885535
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0885535
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0269944, 0.0285305, -0.0522972, 0.0476670
1: -0.0301767, 0.0481623, -0.0381862, 0.0761155, -0.1062922, 0.0863485
2: -0.0622640, 0.0955307, -0.0729797, 0.1120416, -0.1743057, 0.1685104
3: -0.0332325, 0.0495364, -0.0438616, 0.0892202, -0.1224527, 0.0933980
4: -0.0597730, 0.1052151, -0.0815154, 0.1311932, -0.1909662, 0.1867305

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887358, upper bound: 0.0886973
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886108, upper bound: 0.0886464
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886108, upper bound: 0.0886909
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0322836, 0.0334207, -0.0251925, 0.0272970, -0.0595805, 0.0586132
1: -0.0396060, 0.0710703, -0.0353321, 0.0706041, -0.1102102, 0.1064024
2: -0.0933639, 0.1287424, -0.0693398, 0.1112672, -0.2046311, 0.1980822
3: -0.0451630, 0.0779622, -0.0430162, 0.0803674, -0.1255304, 0.1209784
4: -0.0942796, 0.1455511, -0.0769254, 0.1310287, -0.2253083, 0.2224765

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886876, upper bound: 0.0887239
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886108, upper bound: 0.0886726
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886108, upper bound: 0.0888261
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0250651, 0.0229579, -0.0237667, 0.0206726, -0.0457377, 0.0467246
1: -0.0320719, 0.0543695, -0.0301767, 0.0481623, -0.0802343, 0.0845462
2: -0.0693524, 0.1040115, -0.0622640, 0.0955307, -0.1648831, 0.1662756
3: -0.0369494, 0.0586011, -0.0332325, 0.0495364, -0.0864858, 0.0918336
4: -0.0693214, 0.1170426, -0.0597730, 0.1052151, -0.1745365, 0.1768157

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883836, upper bound: 0.0886346
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884127, upper bound: 0.0884950
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884127, upper bound: 0.0886361
time: 0.32 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0261409, 0.0232554, -0.0321830, 0.0320527, -0.0581937, 0.0554384
1: -0.0322643, 0.0522692, -0.0395392, 0.0694895, -0.1017538, 0.0918084
2: -0.0748977, 0.1065080, -0.0931827, 0.1277020, -0.2025998, 0.1996907
3: -0.0383608, 0.0539745, -0.0451072, 0.0761458, -0.1145066, 0.0990818
4: -0.0728056, 0.1203942, -0.0935804, 0.1444632, -0.2172688, 0.2139746

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883061, upper bound: 0.0885881
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885535, upper bound: 0.0884950
time: 0.30 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885535, upper bound: 0.0886361
time: 0.31 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0189038, 0.0158268, -0.0314299, 0.0359206, -0.0548244, 0.0472567
1: -0.0248756, 0.0358514, -0.0441077, 0.0952010, -0.1200766, 0.0799591
2: -0.0468024, 0.0796161, -0.0885184, 0.1337586, -0.1805611, 0.1681345
3: -0.0280160, 0.0339445, -0.0500104, 0.1153530, -0.1433690, 0.0839549
4: -0.0429375, 0.0872777, -0.1020321, 0.1568665, -0.1998040, 0.1893098

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882960, upper bound: 0.0886777
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885356, upper bound: 0.0887088
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885356, upper bound: 0.0887363
time: 0.32 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0274795, 0.0253483, -0.0301374, 0.0339238, -0.0614033, 0.0554857
1: -0.0339634, 0.0554254, -0.0417310, 0.0886000, -0.1225635, 0.0971563
2: -0.0789161, 0.1109374, -0.0860756, 0.1324093, -0.2113254, 0.1970130
3: -0.0402230, 0.0574996, -0.0493154, 0.1056619, -0.1458849, 0.1068150
4: -0.0770320, 0.1258029, -0.0987092, 0.1556965, -0.2327285, 0.2245121

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884397, upper bound: 0.0888167
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885356, upper bound: 0.0887655
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885356, upper bound: 0.0888768
time: 0.31 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0299945, 0.0319951, -0.0557618, 0.0506671
1: -0.0301767, 0.0481623, -0.0375586, 0.0722891, -0.1024658, 0.0857209
2: -0.0622640, 0.0955307, -0.0849874, 0.1250940, -0.1873581, 0.1805181
3: -0.0332325, 0.0495364, -0.0425279, 0.0822176, -0.1154502, 0.0920643
4: -0.0597730, 0.1052151, -0.0872813, 0.1410337, -0.2008067, 0.1924963

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0884192
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0884192
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0322836, 0.0334207, -0.0310192, 0.0308055, -0.0630890, 0.0644399
1: -0.0396060, 0.0710703, -0.0375700, 0.0676793, -0.1072853, 0.1086402
2: -0.0933639, 0.1287424, -0.0899057, 0.1248048, -0.2181687, 0.2186481
3: -0.0451630, 0.0779622, -0.0434730, 0.0742902, -0.1194532, 0.1214352
4: -0.0942796, 0.1455511, -0.0896981, 0.1409828, -0.2352624, 0.2352492

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0885564
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0885564
time: 0.36 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0314299, 0.0359206, -0.0596873, 0.0521025
1: -0.0301767, 0.0481623, -0.0441077, 0.0952010, -0.1253777, 0.0922700
2: -0.0622640, 0.0955307, -0.0885184, 0.1337586, -0.1960227, 0.1840491
3: -0.0332325, 0.0495364, -0.0500104, 0.1153530, -0.1485856, 0.0995468
4: -0.0597730, 0.1052151, -0.1020321, 0.1568665, -0.2166396, 0.2072471

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885353, upper bound: 0.0886575
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886108, upper bound: 0.0887019
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0322836, 0.0334207, -0.0301374, 0.0339238, -0.0662074, 0.0635581
1: -0.0396060, 0.0710703, -0.0417310, 0.0886000, -0.1282061, 0.1128012
2: -0.0933639, 0.1287424, -0.0860756, 0.1324093, -0.2257732, 0.2148180
3: -0.0451630, 0.0779622, -0.0493154, 0.1056619, -0.1508249, 0.1272776
4: -0.0942796, 0.1455511, -0.0987092, 0.1556965, -0.2499761, 0.2442603

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885353, upper bound: 0.0886785
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886108, upper bound: 0.0888293
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0275156, 0.0294149, -0.0189038, 0.0158268, -0.0433424, 0.0483187
1: -0.0389197, 0.0792837, -0.0248756, 0.0358514, -0.0747711, 0.1041593
2: -0.0748440, 0.1147113, -0.0468024, 0.0796161, -0.1544601, 0.1615137
3: -0.0447726, 0.0938445, -0.0280160, 0.0339445, -0.0787171, 0.1218605
4: -0.0845405, 0.1344627, -0.0429375, 0.0872777, -0.1718182, 0.1774001

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886421, upper bound: 0.0882960
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887493, upper bound: 0.0885449
time: 0.39 seconds

## Relational analysis of IS_A2_B1_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887493, upper bound: 0.0885736
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0320182, 0.0394855, -0.0189038, 0.0158268, -0.0478450, 0.0583893
1: -0.0450526, 0.1000605, -0.0248756, 0.0358514, -0.0809040, 0.1249361
2: -0.0900995, 0.1381880, -0.0468024, 0.0796161, -0.1697155, 0.1849904
3: -0.0509707, 0.1209230, -0.0280160, 0.0339445, -0.0849152, 0.1489390
4: -0.1040311, 0.1620654, -0.0429375, 0.0872777, -0.1913089, 0.2050029

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886421, upper bound: 0.0882960
time: 0.39 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887493, upper bound: 0.0885449
time: 0.38 seconds

## Relational analysis of IS_A2_B1_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887493, upper bound: 0.0885736
time: 0.39 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0260446, 0.0281314, -0.0274795, 0.0253483, -0.0513929, 0.0556109
1: -0.0362614, 0.0729389, -0.0339634, 0.0554254, -0.0916867, 0.1069023
2: -0.0718944, 0.1141000, -0.0789161, 0.1109374, -0.1828318, 0.1930161
3: -0.0439528, 0.0835467, -0.0402230, 0.0574996, -0.1014525, 0.1237697
4: -0.0799733, 0.1342807, -0.0770320, 0.1258029, -0.2057763, 0.2113127

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887654, upper bound: 0.0884397
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886463, upper bound: 0.0885353
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886463, upper bound: 0.0886822
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0316295, 0.0377421, -0.0274795, 0.0253483, -0.0569778, 0.0652216
1: -0.0433476, 0.0934690, -0.0339634, 0.0554254, -0.0987730, 0.1274324
2: -0.0898466, 0.1382423, -0.0789161, 0.1109374, -0.2007839, 0.2171584
3: -0.0506773, 0.1110885, -0.0402230, 0.0574996, -0.1081769, 0.1513115
4: -0.1024082, 0.1620971, -0.0770320, 0.1258029, -0.2282111, 0.2391291

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887653, upper bound: 0.0884397
time: 0.34 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886463, upper bound: 0.0885356
time: 0.34 seconds

## Relational analysis of IS_A2_B1_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886463, upper bound: 0.0886822
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0275156, 0.0294149, -0.0237667, 0.0206726, -0.0481882, 0.0531816
1: -0.0389197, 0.0792837, -0.0301767, 0.0481623, -0.0870821, 0.1094604
2: -0.0748440, 0.1147113, -0.0622640, 0.0955307, -0.1703747, 0.1769753
3: -0.0447726, 0.0938445, -0.0332325, 0.0495364, -0.0943090, 0.1270771
4: -0.0845405, 0.1344627, -0.0597730, 0.1052151, -0.1897556, 0.1942357

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886973, upper bound: 0.0887358
time: 0.36 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886463, upper bound: 0.0886108
time: 0.35 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886463, upper bound: 0.0887371
time: 0.36 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0260446, 0.0281314, -0.0322836, 0.0334207, -0.0594653, 0.0604150
1: -0.0362614, 0.0729389, -0.0396060, 0.0710703, -0.1073316, 0.1125450
2: -0.0718944, 0.1141000, -0.0933639, 0.1287424, -0.2006368, 0.2074639
3: -0.0439528, 0.0835467, -0.0451630, 0.0779622, -0.1219150, 0.1287097
4: -0.0799733, 0.1342807, -0.0942796, 0.1455511, -0.2255244, 0.2285603

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887239, upper bound: 0.0888083
time: 0.37 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886726, upper bound: 0.0886108
time: 0.34 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886726, upper bound: 0.0888125
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0320182, 0.0394855, -0.0237667, 0.0206726, -0.0526908, 0.0632522
1: -0.0450526, 0.1000605, -0.0301767, 0.0481623, -0.0932150, 0.1302372
2: -0.0900995, 0.1381880, -0.0622640, 0.0955307, -0.1856302, 0.2004520
3: -0.0509707, 0.1209230, -0.0332325, 0.0495364, -0.1005071, 0.1541555
4: -0.1040311, 0.1620654, -0.0597730, 0.1052151, -0.2092462, 0.2218385

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887088, upper bound: 0.0885455
time: 0.32 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886463, upper bound: 0.0885790
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0316295, 0.0377421, -0.0322836, 0.0334207, -0.0650501, 0.0700257
1: -0.0433476, 0.0934690, -0.0396060, 0.0710703, -0.1144179, 0.1330750
2: -0.0898466, 0.1382423, -0.0933639, 0.1287424, -0.2185889, 0.2316062
3: -0.0506773, 0.1110885, -0.0451630, 0.0779622, -0.1286395, 0.1562514
4: -0.1024082, 0.1620971, -0.0942796, 0.1455511, -0.2479593, 0.2563767

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886726, upper bound: 0.0885455
time: 0.33 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887652, upper bound: 0.0886825
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0281139, 0.0310250, -0.0358490, 0.0487337, -0.0768476, 0.0668740
1: -0.0406024, 0.0818455, -0.0507856, 0.1154432, -0.1560455, 0.1326310
2: -0.0773617, 0.1205507, -0.1024819, 0.1582004, -0.2355621, 0.2230326
3: -0.0474901, 0.0966178, -0.0580275, 0.1404007, -0.1878907, 0.1546452
4: -0.0884082, 0.1426494, -0.1195223, 0.1871233, -0.2755313, 0.2621717

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886069, upper bound: 0.0881611
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886069, upper bound: 0.0882075
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0200956, 0.0350137, -0.0358490, 0.0487337, -0.0688294, 0.0708628
1: -0.0474771, 0.1145338, -0.0507856, 0.1154432, -0.1629202, 0.1653194
2: -0.0544097, 0.1152232, -0.1024819, 0.1582004, -0.2126101, 0.2177052
3: -0.0536702, 0.1447884, -0.0580275, 0.1404007, -0.1940709, 0.2028158
4: -0.0832517, 0.1436815, -0.1195223, 0.1871233, -0.2703749, 0.2632038

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878932, upper bound: 0.0876571
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878917, upper bound: 0.0876674
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0281048, 0.0310157, -0.0262249, 0.0476892, -0.0757940, 0.0572406
1: -0.0405944, 0.0818225, -0.0560731, 0.1461119, -0.1867063, 0.1378957
2: -0.0773178, 0.1205083, -0.0741902, 0.1451171, -0.2224349, 0.1946985
3: -0.0474758, 0.0965890, -0.0633892, 0.1894719, -0.2369477, 0.1599782
4: -0.0883616, 0.1425957, -0.1138687, 0.1800817, -0.2684433, 0.2564643

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885452, upper bound: 0.0881703
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885666, upper bound: 0.0881785
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0200956, 0.0350137, -0.0262249, 0.0476892, -0.0677848, 0.0612386
1: -0.0474771, 0.1145338, -0.0560731, 0.1461119, -0.1935890, 0.1706070
2: -0.0544097, 0.1152232, -0.0741902, 0.1451171, -0.1995267, 0.1894134
3: -0.0536702, 0.1447884, -0.0633892, 0.1894719, -0.2431421, 0.2081776
4: -0.0832517, 0.1436815, -0.1138687, 0.1800817, -0.2633333, 0.2575501

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879677, upper bound: 0.0879167
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879677, upper bound: 0.0879167
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0326413, 0.0406024, -0.0358490, 0.0487337, -0.0813750, 0.0764514
1: -0.0464028, 0.1029233, -0.0507856, 0.1154432, -0.1618460, 0.1537088
2: -0.0931736, 0.1439410, -0.1024819, 0.1582004, -0.2513740, 0.2464229
3: -0.0537068, 0.1251042, -0.0580275, 0.1404007, -0.1941074, 0.1831317
4: -0.1092366, 0.1700594, -0.1195223, 0.1871233, -0.2963600, 0.2895818

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893313, upper bound: 0.0888084
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0893313, upper bound: 0.0888084
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0252827, 0.0445866, -0.0358490, 0.0487337, -0.0740165, 0.0804356
1: -0.0539653, 0.1393512, -0.0507856, 0.1154432, -0.1694084, 0.1901368
2: -0.0712305, 0.1388832, -0.1024819, 0.1582004, -0.2294308, 0.2413651
3: -0.0606718, 0.1808298, -0.0580275, 0.1404007, -0.2010725, 0.2388572
4: -0.1092791, 0.1714957, -0.1195223, 0.1871233, -0.2964023, 0.2910181

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894665, upper bound: 0.0888260
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0894551, upper bound: 0.0888373
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0320116, 0.0393650, -0.0262249, 0.0476892, -0.0797008, 0.0655898
1: -0.0457655, 0.1012265, -0.0560731, 0.1461119, -0.1918774, 0.1572997
2: -0.0914162, 0.1418947, -0.0741902, 0.1451171, -0.2365332, 0.2160849
3: -0.0532144, 0.1229988, -0.0633892, 0.1894719, -0.2426863, 0.1863880
4: -0.1074655, 0.1679001, -0.1138687, 0.1800817, -0.2875472, 0.2817687

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 8

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887795, upper bound: 0.0887843
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0252827, 0.0445866, -0.0262249, 0.0476892, -0.0729719, 0.0708114
1: -0.0539653, 0.1393512, -0.0560731, 0.1461119, -0.2000772, 0.1954244
2: -0.0712305, 0.1388832, -0.0741902, 0.1451171, -0.2163476, 0.2130734
3: -0.0606718, 0.1808298, -0.0633892, 0.1894719, -0.2501436, 0.2442190
4: -0.1092791, 0.1714957, -0.1138687, 0.1800817, -0.2893607, 0.2853644

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879106, upper bound: 0.0879721
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880213
time: 0.36 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.88 seconds
IS_A1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0893074, upper bound: 0.0893074
IS_A1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0893074, upper bound: 0.0893074
IS_A1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0893074, upper bound: 0.0893074
IS_A1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0893074, upper bound: 0.0893074
IS_A1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0894212, upper bound: 0.0895949
IS_A1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0894212, upper bound: 0.0895949
IS_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0894212, upper bound: 0.0895949
IS_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0894212, upper bound: 0.0895949
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0884127
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0884127
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0884127, upper bound: 0.0885535
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0885535
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0886108, upper bound: 0.0886464
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0886108, upper bound: 0.0886909
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0886108, upper bound: 0.0886726
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0886108, upper bound: 0.0888261
IS_A1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0884127, upper bound: 0.0884950
IS_A1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0884127, upper bound: 0.0886361
IS_A1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0885535, upper bound: 0.0884950
IS_A1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0885535, upper bound: 0.0886361
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0885356, upper bound: 0.0887088
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0885356, upper bound: 0.0887363
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0885356, upper bound: 0.0887655
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0885356, upper bound: 0.0888768
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0884192
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0884192
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0885564
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0884950, upper bound: 0.0885564
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0885353, upper bound: 0.0886575
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0886108, upper bound: 0.0887019
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0885353, upper bound: 0.0886785
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0886108, upper bound: 0.0888293
IS_A2_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0887493, upper bound: 0.0885449
IS_A2_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0887493, upper bound: 0.0885736
IS_A2_B1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0887493, upper bound: 0.0885449
IS_A2_B1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0887493, upper bound: 0.0885736
IS_A2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0886463, upper bound: 0.0885353
IS_A2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0886463, upper bound: 0.0886822
IS_A2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0886463, upper bound: 0.0885356
IS_A2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0886463, upper bound: 0.0886822
IS_A2_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0886463, upper bound: 0.0886108
IS_A2_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0886463, upper bound: 0.0887371
IS_A2_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0886726, upper bound: 0.0886108
IS_A2_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0886726, upper bound: 0.0888125
IS_A2_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0887088, upper bound: 0.0885455
IS_A2_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0886463, upper bound: 0.0885790
IS_A2_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0886726, upper bound: 0.0885455
IS_A2_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0887652, upper bound: 0.0886825
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0886069, upper bound: 0.0881611
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0886069, upper bound: 0.0882075
IS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0878932, upper bound: 0.0876571
IS_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0878917, upper bound: 0.0876674
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0885452, upper bound: 0.0881703
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0885666, upper bound: 0.0881785
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0879677, upper bound: 0.0879167
IS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0879677, upper bound: 0.0879167
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0893313, upper bound: 0.0888084
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0893313, upper bound: 0.0888084
IS_A2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0894665, upper bound: 0.0888260
IS_A2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0894551, upper bound: 0.0888373
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0887795, upper bound: 0.0887843
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0887910, upper bound: 0.0887871
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0879106, upper bound: 0.0879721
IS_A2_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.88
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880213

## BFS IS instance: IS_A1_B1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0271122, 0.0295338, -0.0263475, 0.0250260, -0.0521382, 0.0558814
1: -0.0338762, 0.0683478, -0.0333350, 0.0601690, -0.0940453, 0.1016828
2: -0.0775269, 0.1190995, -0.0742820, 0.1107289, -0.1882558, 0.1933815
3: -0.0413828, 0.0783211, -0.0402154, 0.0662137, -0.1075965, 0.1185365
4: -0.0817343, 0.1358244, -0.0753473, 0.1262361, -0.2079704, 0.2111717

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882992, upper bound: 0.0885008
time: 0.42 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883064, upper bound: 0.0885059
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0280932, 0.0294513, -0.0263475, 0.0250260, -0.0531193, 0.0557989
1: -0.0348121, 0.0680492, -0.0333350, 0.0601690, -0.0949811, 0.1013842
2: -0.0808951, 0.1180537, -0.0742820, 0.1107289, -0.1916240, 0.1923357
3: -0.0416559, 0.0777593, -0.0402154, 0.0662137, -0.1078696, 0.1179747
4: -0.0845517, 0.1341558, -0.0753473, 0.1262361, -0.2107877, 0.2095031

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882992, upper bound: 0.0885008
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883064, upper bound: 0.0885059
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0271122, 0.0295338, -0.0274014, 0.0255119, -0.0526241, 0.0569353
1: -0.0338762, 0.0683478, -0.0342971, 0.0605227, -0.0943990, 0.1026449
2: -0.0775269, 0.1190995, -0.0780043, 0.1108751, -0.1884020, 0.1971038
3: -0.0413828, 0.0783211, -0.0405568, 0.0666638, -0.1080466, 0.1188779
4: -0.0817343, 0.1358244, -0.0788346, 0.1258900, -0.2076243, 0.2146589

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882690, upper bound: 0.0882802
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882591, upper bound: 0.0882591
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0280932, 0.0294513, -0.0274014, 0.0255119, -0.0536052, 0.0568528
1: -0.0348121, 0.0680492, -0.0342971, 0.0605227, -0.0953348, 0.1023463
2: -0.0808951, 0.1180537, -0.0780043, 0.1108751, -0.1917702, 0.1960580
3: -0.0416559, 0.0777593, -0.0405568, 0.0666638, -0.1083197, 0.1183161
4: -0.0845517, 0.1341558, -0.0788346, 0.1258900, -0.2104416, 0.2129904

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882690, upper bound: 0.0882802
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882591, upper bound: 0.0882591
time: 0.32 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0271122, 0.0295338, -0.0234641, 0.0292775, -0.0563897, 0.0529980
1: -0.0338762, 0.0683478, -0.0357470, 0.0820802, -0.1159564, 0.1040948
2: -0.0775269, 0.1190995, -0.0645553, 0.1123769, -0.1899039, 0.1836547
3: -0.0413828, 0.0783211, -0.0443082, 0.1001011, -0.1414839, 0.1226293
4: -0.0817343, 0.1358244, -0.0799981, 0.1339988, -0.2157331, 0.2158224

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883023, upper bound: 0.0886421
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884456, upper bound: 0.0887654
time: 0.37 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0280932, 0.0294513, -0.0234641, 0.0292775, -0.0573707, 0.0529155
1: -0.0348121, 0.0680492, -0.0357470, 0.0820802, -0.1168922, 0.1037963
2: -0.0808951, 0.1180537, -0.0645553, 0.1123769, -0.1932720, 0.1826089
3: -0.0416559, 0.0777593, -0.0443082, 0.1001011, -0.1417570, 0.1220675
4: -0.0845517, 0.1341558, -0.0799981, 0.1339988, -0.2185504, 0.2141538

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883023, upper bound: 0.0886421
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884456, upper bound: 0.0887654
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0271122, 0.0295338, -0.0287184, 0.0323664, -0.0594786, 0.0582523
1: -0.0338762, 0.0683478, -0.0403162, 0.0872710, -0.1211473, 0.1086640
2: -0.0775269, 0.1190995, -0.0803196, 0.1253408, -0.2028677, 0.1994191
3: -0.0413828, 0.0783211, -0.0482766, 0.1049064, -0.1462892, 0.1265977
4: -0.0817343, 0.1358244, -0.0931064, 0.1482157, -0.2299500, 0.2289308

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882989, upper bound: 0.0885860
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884456, upper bound: 0.0886706
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0280932, 0.0294513, -0.0287184, 0.0323664, -0.0604596, 0.0581698
1: -0.0348121, 0.0680492, -0.0403162, 0.0872710, -0.1220831, 0.1083654
2: -0.0808951, 0.1180537, -0.0803196, 0.1253408, -0.2062359, 0.1983733
3: -0.0416559, 0.0777593, -0.0482766, 0.1049064, -0.1465623, 0.1260359
4: -0.0845517, 0.1341558, -0.0931064, 0.1482157, -0.2327674, 0.2272622

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0882989, upper bound: 0.0885860
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884456, upper bound: 0.0886706
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0183703, 0.0149739, -0.0387406, 0.0390429
1: -0.0301767, 0.0481623, -0.0246088, 0.0324230, -0.0625997, 0.0727711
2: -0.0622640, 0.0955307, -0.0432060, 0.0750874, -0.1373515, 0.1387367
3: -0.0332325, 0.0495364, -0.0270328, 0.0289099, -0.0621424, 0.0765692
4: -0.0597730, 0.1052151, -0.0385889, 0.0818611, -0.1416342, 0.1438040

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885321, upper bound: 0.0883836
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 21

Time for candidate selection: 2.15 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 49
type: B, layer: 5, pos: 49
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 33
type: B, layer: 5, pos: 21
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 21
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 23
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 16
type: B, layer: 5, pos: 42
type: A, layer: 5, pos: 1
type: A, layer: 5, pos: 42
type: B, layer: 5, pos: 23
type: B, layer: 5, pos: 26
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 14
type: A, layer: 5, pos: 16

Time for candidate selection: 5.61 seconds

### Candidate
type: A, layer: 5, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 49

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0864627, upper bound: 0.0870426
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880758, upper bound: 0.0880584
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0269960, 0.0243486, -0.0481153, 0.0476686
1: -0.0301767, 0.0481623, -0.0335704, 0.0519866, -0.0821633, 0.0817327
2: -0.0622640, 0.0955307, -0.0763385, 0.1069899, -0.1692539, 0.1718692
3: -0.0332325, 0.0495364, -0.0392262, 0.0524801, -0.0857127, 0.0887626
4: -0.0597730, 0.1052151, -0.0729759, 0.1209708, -0.1807439, 0.1781910

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0885321, upper bound: 0.0883836
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36

Time for candidate selection: 2.17 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 33
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 23
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 23
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 16
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 26
type: A, layer: 5, pos: 49
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 1
type: A, layer: 5, pos: 16
type: A, layer: 5, pos: 14
type: A, layer: 5, pos: 42

Time for candidate selection: 5.81 seconds

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0864627, upper bound: 0.0870426
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880758, upper bound: 0.0880584
time: 0.34 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0322836, 0.0334207, -0.0183703, 0.0149739, -0.0472575, 0.0517910
1: -0.0396060, 0.0710703, -0.0246088, 0.0324230, -0.0720291, 0.0956790
2: -0.0933639, 0.1287424, -0.0432060, 0.0750874, -0.1684513, 0.1719484
3: -0.0451630, 0.0779622, -0.0270328, 0.0289099, -0.0740729, 0.1049950
4: -0.0942796, 0.1455511, -0.0385889, 0.0818611, -0.1761408, 0.1841400

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0883486, upper bound: 0.0882964
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 21

Time for candidate selection: 2.30 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 33
type: B, layer: 5, pos: 26
type: A, layer: 5, pos: 26
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 23
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 34
type: A, layer: 5, pos: 16
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 1
type: B, layer: 5, pos: 23
type: B, layer: 5, pos: 49
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 28
type: B, layer: 5, pos: 42

Time for candidate selection: 6.03 seconds

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0863725, upper bound: 0.0870174
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879620, upper bound: 0.0879925
time: 0.33 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0322836, 0.0334207, -0.0269960, 0.0243486, -0.0566321, 0.0604167
1: -0.0396060, 0.0710703, -0.0335704, 0.0519866, -0.0915926, 0.1046406
2: -0.0933639, 0.1287424, -0.0763385, 0.1069899, -0.2003538, 0.2050809
3: -0.0451630, 0.0779622, -0.0392262, 0.0524801, -0.0976431, 0.1171884
4: -0.0942796, 0.1455511, -0.0729759, 0.1209708, -0.2152504, 0.2185270

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0884308, upper bound: 0.0883061
time: 0.36 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 21

Time for candidate selection: 2.35 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 33
type: B, layer: 5, pos: 33
type: B, layer: 5, pos: 26
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 21
type: A, layer: 5, pos: 26
type: A, layer: 5, pos: 23
type: B, layer: 5, pos: 23
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 15
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 28
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 16
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 1

Time for candidate selection: 6.06 seconds

### Candidate
type: A, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 33

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 26

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0863725, upper bound: 0.0870174
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879620, upper bound: 0.0879938
time: 0.35 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0219219, 0.0215290, -0.0452957, 0.0425945
1: -0.0301767, 0.0481623, -0.0300346, 0.0547859, -0.0849626, 0.0781969
2: -0.0622640, 0.0955307, -0.0547445, 0.0888233, -0.1510873, 0.1502752
3: -0.0332325, 0.0495364, -0.0346989, 0.0597394, -0.0929720, 0.0842352
4: -0.0597730, 0.1052151, -0.0569638, 0.1019325, -0.1617055, 0.1621789

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886435, upper bound: 0.0886973
time: 0.37 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 16

Time for candidate selection: 2.31 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 35

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886522, upper bound: 0.0887477
time: 0.35 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 49
type: B, layer: 5, pos: 49
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 33
type: A, layer: 5, pos: 23
type: A, layer: 5, pos: 21
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 24
type: B, layer: 5, pos: 23
type: B, layer: 5, pos: 15
type: B, layer: 5, pos: 26
type: B, layer: 5, pos: 34
type: B, layer: 5, pos: 1
type: A, layer: 5, pos: 1
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 16
type: A, layer: 5, pos: 14
type: A, layer: 5, pos: 16
type: A, layer: 5, pos: 42
type: B, layer: 5, pos: 48
type: B, layer: 5, pos: 2

Time for candidate selection: 7.56 seconds

### Candidate
type: A, layer: 5, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 49

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0863831, upper bound: 0.0870049
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0881283, upper bound: 0.0882694
time: 0.36 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0237667, 0.0206726, -0.0263492, 0.0286028, -0.0523695, 0.0470219
1: -0.0301767, 0.0481623, -0.0374302, 0.0718135, -0.1019902, 0.0855925
2: -0.0622640, 0.0955307, -0.0725907, 0.1131533, -0.1754173, 0.1681214
3: -0.0332325, 0.0495364, -0.0441836, 0.0810009, -0.1142334, 0.0937200
4: -0.0597730, 0.1052151, -0.0796982, 0.1329616, -0.1927346, 0.1849132

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0886435, upper bound: 0.0886973
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 16
type: B, layer: 3, pos: 36
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 36
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 16

Time for candidate selection: 2.14 seconds

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 36

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0250000, mid=0.0250000, abs_max=0.10251016169786453
rel_dist={0: [-0.08996129100877424, 0.08996129100877422]}

## Binary Search with IS_dual Result
status: None
Maximum delta epsilon: None
execution time: 1151.46 seconds
