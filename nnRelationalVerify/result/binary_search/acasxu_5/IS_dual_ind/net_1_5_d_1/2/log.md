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
execution time: IAR + LP analysis = 1.60 + 0.90 = 2.50 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0899877, upper bound: 0.0899877


# Binary Search by BASE starts (time budget: 1197.50 seconds, max iter: 100)

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
Binary search time: 47.29 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1150.21 seconds

## Binary search (step 0) starts
Candidate diff: 0.1000000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895312, upper bound: 0.0896540
time: 0.32 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781
time: 0.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.78 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -0.0895312, upper bound: 0.0896540
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0390887, 0.0634215, -0.0938137, 0.0706923
1: -0.0419247, 0.0834490, -0.0557757, 0.1414140, -0.1833387, 0.1392248
2: -0.0824195, 0.1223470, -0.1134923, 0.1820314, -0.2644509, 0.2358393
3: -0.0466005, 0.0980623, -0.0636387, 0.1747649, -0.2213654, 0.1617010
4: -0.0908781, 0.1427428, -0.1363116, 0.2151742, -0.3060522, 0.2790544

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
time: 0.30 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
time: 0.32 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0383779, 0.0609533, -0.0968511, 0.0905146
1: -0.0510239, 0.1157805, -0.0546976, 0.1366976, -0.1877214, 0.1704781
2: -0.1029525, 0.1605961, -0.1109459, 0.1778313, -0.2807838, 0.2715421
3: -0.0576805, 0.1386453, -0.0624058, 0.1685225, -0.2262030, 0.2010511
4: -0.1189080, 0.1891569, -0.1327503, 0.2101621, -0.3290701, 0.3219072

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0892359, upper bound: 0.0882037
time: 0.32 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880320
time: 0.32 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.27 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.0892359, upper bound: 0.0882037
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880320

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0303923, 0.0316036, -0.0619959, 0.0619959
1: -0.0419247, 0.0834490, -0.0419247, 0.0834490, -0.1253737, 0.1253737
2: -0.0824195, 0.1223470, -0.0824195, 0.1223470, -0.2047665, 0.2047665
3: -0.0466005, 0.0980623, -0.0466005, 0.0980623, -0.1446628, 0.1446628
4: -0.0908781, 0.1427428, -0.0908781, 0.1427428, -0.2336209, 0.2336209

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880314, upper bound: 0.0890560
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
time: 0.31 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0358978, 0.0521367, -0.0825290, 0.0675014
1: -0.0419247, 0.0834490, -0.0510239, 0.1157805, -0.1577052, 0.1344729
2: -0.0824195, 0.1223470, -0.1029525, 0.1605961, -0.2430156, 0.2252996
3: -0.0466005, 0.0980623, -0.0576805, 0.1386453, -0.1852457, 0.1557429
4: -0.0908781, 0.1427428, -0.1189080, 0.1891569, -0.2800349, 0.2616507

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880314, upper bound: 0.0890562
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
time: 0.29 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0358793, 0.0494104, -0.0853081, 0.0880160
1: -0.0510239, 0.1157805, -0.0512398, 0.1172707, -0.1682945, 0.1670202
2: -0.1029525, 0.1605961, -0.1024367, 0.1594433, -0.2623959, 0.2630328
3: -0.0576805, 0.1386453, -0.0584664, 0.1428694, -0.2005499, 0.1971117
4: -0.1189080, 0.1891569, -0.1198549, 0.1886917, -0.3075997, 0.3090118

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880320
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880320
time: 0.38 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.32 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.0880314, upper bound: 0.0890560
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.0880314, upper bound: 0.0890562
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880320
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880320

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0303923, 0.0316036, -0.0601504, 0.0592287
1: -0.0394259, 0.0744049, -0.0419247, 0.0834490, -0.1228749, 0.1163296
2: -0.0761525, 0.1130898, -0.0824195, 0.1223470, -0.1984996, 0.1955093
3: -0.0434286, 0.0854480, -0.0466005, 0.0980623, -0.1414909, 0.1320485
4: -0.0817791, 0.1311970, -0.0908781, 0.1427428, -0.2245218, 0.2220750

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0358978, 0.0521367, -0.0806835, 0.0647341
1: -0.0394259, 0.0744049, -0.0510239, 0.1157805, -0.1552063, 0.1254288
2: -0.0761525, 0.1130898, -0.1029525, 0.1605961, -0.2367487, 0.2160423
3: -0.0434286, 0.0854480, -0.0576805, 0.1386453, -0.1820739, 0.1431285
4: -0.0817791, 0.1311970, -0.1189080, 0.1891569, -0.2709360, 0.2501049

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.32 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.23 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.23
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.23
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.23
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.23
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
Binary search (step 0): status=Status.VERIFIED, low=0.1000000, high=0.2000000, mid=0.1000000, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 1) starts
Candidate diff: 0.1500000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895480, upper bound: 0.0897708
time: 0.31 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781
time: 0.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.77 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -0.0895480, upper bound: 0.0897708
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0390887, 0.0634215, -0.0938137, 0.0706923
1: -0.0419247, 0.0834490, -0.0557757, 0.1414140, -0.1833387, 0.1392248
2: -0.0824195, 0.1223470, -0.1134923, 0.1820314, -0.2644509, 0.2358393
3: -0.0466005, 0.0980623, -0.0636387, 0.1747649, -0.2213654, 0.1617010
4: -0.0908781, 0.1427428, -0.1363116, 0.2151742, -0.3060522, 0.2790544

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
time: 0.33 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
time: 0.30 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0390887, 0.0634215, -0.0993192, 0.0912254
1: -0.0510239, 0.1157805, -0.0557757, 0.1414140, -0.1924379, 0.1715562
2: -0.1029525, 0.1605961, -0.1134923, 0.1820314, -0.2849840, 0.2740884
3: -0.0576805, 0.1386453, -0.0636387, 0.1747649, -0.2324454, 0.2022839
4: -0.1189080, 0.1891569, -0.1363116, 0.2151742, -0.3340822, 0.3254685

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
time: 0.33 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781
time: 0.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.22 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.22
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0303923, 0.0316036, -0.0619959, 0.0619959
1: -0.0419247, 0.0834490, -0.0419247, 0.0834490, -0.1253737, 0.1253737
2: -0.0824195, 0.1223470, -0.0824195, 0.1223470, -0.2047665, 0.2047665
3: -0.0466005, 0.0980623, -0.0466005, 0.0980623, -0.1446628, 0.1446628
4: -0.0908781, 0.1427428, -0.0908781, 0.1427428, -0.2336209, 0.2336209

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880444, upper bound: 0.0893040
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
time: 0.31 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0358978, 0.0521367, -0.0825290, 0.0675014
1: -0.0419247, 0.0834490, -0.0510239, 0.1157805, -0.1577052, 0.1344729
2: -0.0824195, 0.1223470, -0.1029525, 0.1605961, -0.2430156, 0.2252996
3: -0.0466005, 0.0980623, -0.0576805, 0.1386453, -0.1852457, 0.1557429
4: -0.0908781, 0.1427428, -0.1189080, 0.1891569, -0.2800349, 0.2616507

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880444, upper bound: 0.0893056
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
time: 0.32 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0303923, 0.0316036, -0.0675014, 0.0825290
1: -0.0510239, 0.1157805, -0.0419247, 0.0834490, -0.1344729, 0.1577052
2: -0.1029525, 0.1605961, -0.0824195, 0.1223470, -0.2252996, 0.2430156
3: -0.0576805, 0.1386453, -0.0466005, 0.0980623, -0.1557429, 0.1852458
4: -0.1189080, 0.1891569, -0.0908781, 0.1427428, -0.2616507, 0.2800350

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0892936
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
time: 0.34 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0358978, 0.0521367, -0.0880345, 0.0880345
1: -0.0510239, 0.1157805, -0.0510239, 0.1157805, -0.1668044, 0.1668043
2: -0.1029525, 0.1605961, -0.1029525, 0.1605961, -0.2635486, 0.2635487
3: -0.0576805, 0.1386453, -0.0576805, 0.1386453, -0.1963258, 0.1963258
4: -0.1189080, 0.1891569, -0.1189080, 0.1891569, -0.3080648, 0.3080649

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893464
time: 0.32 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317
time: 0.33 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.52 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0880444, upper bound: 0.0893040
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0880444, upper bound: 0.0893056
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0892936
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893464
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0303923, 0.0316036, -0.0601504, 0.0592287
1: -0.0394259, 0.0744049, -0.0419247, 0.0834490, -0.1228749, 0.1163296
2: -0.0761525, 0.1130898, -0.0824195, 0.1223470, -0.1984996, 0.1955093
3: -0.0434286, 0.0854480, -0.0466005, 0.0980623, -0.1414909, 0.1320485
4: -0.0817791, 0.1311970, -0.0908781, 0.1427428, -0.2245218, 0.2220750

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.32 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0358978, 0.0521367, -0.0806835, 0.0647341
1: -0.0394259, 0.0744049, -0.0510239, 0.1157805, -0.1552063, 0.1254288
2: -0.0761525, 0.1130898, -0.1029525, 0.1605961, -0.2367487, 0.2160423
3: -0.0434286, 0.0854480, -0.0576805, 0.1386453, -0.1820739, 0.1431285
4: -0.0817791, 0.1311970, -0.1189080, 0.1891569, -0.2709360, 0.2501049

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.31 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0303923, 0.0316036, -0.0634249, 0.0704445
1: -0.0462006, 0.0969325, -0.0419247, 0.0834490, -0.1296496, 0.1388572
2: -0.0906157, 0.1402107, -0.0824195, 0.1223470, -0.2129627, 0.2226302
3: -0.0531606, 0.1144974, -0.0466005, 0.0980623, -0.1512229, 0.1610979
4: -0.1036385, 0.1661279, -0.0908781, 0.1427428, -0.2463813, 0.2570060

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0358978, 0.0521367, -0.0839581, 0.0759500
1: -0.0462006, 0.0969325, -0.0510239, 0.1157805, -0.1619810, 0.1479563
2: -0.0906157, 0.1402107, -0.1029525, 0.1605961, -0.2512118, 0.2431632
3: -0.0531606, 0.1144974, -0.0576805, 0.1386453, -0.1918058, 0.1721780
4: -0.1036385, 0.1661279, -0.1189080, 0.1891569, -0.2927953, 0.2850358

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.35 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.58 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
Binary search (step 1): status=Status.VERIFIED, low=0.1500000, high=0.2000000, mid=0.1500000, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 2) starts
Candidate diff: 0.1750000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895551, upper bound: 0.0897790
time: 0.32 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781
time: 0.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.79 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -0.0895551, upper bound: 0.0897790
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0390887, 0.0634215, -0.0938137, 0.0706923
1: -0.0419247, 0.0834490, -0.0557757, 0.1414140, -0.1833387, 0.1392248
2: -0.0824195, 0.1223470, -0.1134923, 0.1820314, -0.2644509, 0.2358393
3: -0.0466005, 0.0980623, -0.0636387, 0.1747649, -0.2213654, 0.1617010
4: -0.0908781, 0.1427428, -0.1363116, 0.2151742, -0.3060522, 0.2790544

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
time: 0.32 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
time: 0.32 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0390887, 0.0634215, -0.0993192, 0.0912254
1: -0.0510239, 0.1157805, -0.0557757, 0.1414140, -0.1924379, 0.1715562
2: -0.1029525, 0.1605961, -0.1134923, 0.1820314, -0.2849840, 0.2740884
3: -0.0576805, 0.1386453, -0.0636387, 0.1747649, -0.2324454, 0.2022839
4: -0.1189080, 0.1891569, -0.1363116, 0.2151742, -0.3340822, 0.3254685

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
time: 0.31 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781
time: 0.31 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.46 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.46
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.46
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.46
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.46
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0303923, 0.0316036, -0.0619959, 0.0619959
1: -0.0419247, 0.0834490, -0.0419247, 0.0834490, -0.1253737, 0.1253737
2: -0.0824195, 0.1223470, -0.0824195, 0.1223470, -0.2047665, 0.2047665
3: -0.0466005, 0.0980623, -0.0466005, 0.0980623, -0.1446628, 0.1446628
4: -0.0908781, 0.1427428, -0.0908781, 0.1427428, -0.2336209, 0.2336209

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880493, upper bound: 0.0894219
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
time: 0.32 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0358978, 0.0521367, -0.0825290, 0.0675014
1: -0.0419247, 0.0834490, -0.0510239, 0.1157805, -0.1577052, 0.1344729
2: -0.0824195, 0.1223470, -0.1029525, 0.1605961, -0.2430156, 0.2252996
3: -0.0466005, 0.0980623, -0.0576805, 0.1386453, -0.1852457, 0.1557429
4: -0.0908781, 0.1427428, -0.1189080, 0.1891569, -0.2800349, 0.2616507

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880493, upper bound: 0.0894265
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
time: 0.30 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0303923, 0.0316036, -0.0675014, 0.0825290
1: -0.0510239, 0.1157805, -0.0419247, 0.0834490, -0.1344729, 0.1577052
2: -0.1029525, 0.1605961, -0.0824195, 0.1223470, -0.2252996, 0.2430156
3: -0.0576805, 0.1386453, -0.0466005, 0.0980623, -0.1557429, 0.1852458
4: -0.1189080, 0.1891569, -0.0908781, 0.1427428, -0.2616507, 0.2800350

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
time: 0.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0358978, 0.0521367, -0.0880345, 0.0880345
1: -0.0510239, 0.1157805, -0.0510239, 0.1157805, -0.1668044, 0.1668043
2: -0.1029525, 0.1605961, -0.1029525, 0.1605961, -0.2635486, 0.2635487
3: -0.0576805, 0.1386453, -0.0576805, 0.1386453, -0.1963258, 0.1963258
4: -0.1189080, 0.1891569, -0.1189080, 0.1891569, -0.3080648, 0.3080649

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
time: 0.32 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317
time: 0.33 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.36 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.36
Output dim: 0, lower bound: -0.0880493, upper bound: 0.0894219
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.36
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.36
Output dim: 0, lower bound: -0.0880493, upper bound: 0.0894265
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.36
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.36
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.36
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.36
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.36
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0303923, 0.0316036, -0.0601504, 0.0592287
1: -0.0394259, 0.0744049, -0.0419247, 0.0834490, -0.1228749, 0.1163296
2: -0.0761525, 0.1130898, -0.0824195, 0.1223470, -0.1984996, 0.1955093
3: -0.0434286, 0.0854480, -0.0466005, 0.0980623, -0.1414909, 0.1320485
4: -0.0817791, 0.1311970, -0.0908781, 0.1427428, -0.2245218, 0.2220750

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0358978, 0.0521367, -0.0806835, 0.0647341
1: -0.0394259, 0.0744049, -0.0510239, 0.1157805, -0.1552063, 0.1254288
2: -0.0761525, 0.1130898, -0.1029525, 0.1605961, -0.2367487, 0.2160423
3: -0.0434286, 0.0854480, -0.0576805, 0.1386453, -0.1820739, 0.1431285
4: -0.0817791, 0.1311970, -0.1189080, 0.1891569, -0.2709360, 0.2501049

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0303923, 0.0316036, -0.0634249, 0.0704445
1: -0.0462006, 0.0969325, -0.0419247, 0.0834490, -0.1296496, 0.1388572
2: -0.0906157, 0.1402107, -0.0824195, 0.1223470, -0.2129627, 0.2226302
3: -0.0531606, 0.1144974, -0.0466005, 0.0980623, -0.1512229, 0.1610979
4: -0.1036385, 0.1661279, -0.0908781, 0.1427428, -0.2463813, 0.2570060

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0358978, 0.0521367, -0.0839581, 0.0759500
1: -0.0462006, 0.0969325, -0.0510239, 0.1157805, -0.1619810, 0.1479563
2: -0.0906157, 0.1402107, -0.1029525, 0.1605961, -0.2512118, 0.2431632
3: -0.0531606, 0.1144974, -0.0576805, 0.1386453, -0.1918058, 0.1721780
4: -0.1036385, 0.1661279, -0.1189080, 0.1891569, -0.2927953, 0.2850358

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.33 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.42 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.42
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
Binary search (step 2): status=Status.VERIFIED, low=0.1750000, high=0.2000000, mid=0.1750000, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 3) starts
Candidate diff: 0.1875000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895581, upper bound: 0.0897790
time: 0.33 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781
time: 0.33 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.81 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.81
Output dim: 0, lower bound: -0.0895581, upper bound: 0.0897790
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.81
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0390887, 0.0634215, -0.0938137, 0.0706923
1: -0.0419247, 0.0834490, -0.0557757, 0.1414140, -0.1833387, 0.1392248
2: -0.0824195, 0.1223470, -0.1134923, 0.1820314, -0.2644509, 0.2358393
3: -0.0466005, 0.0980623, -0.0636387, 0.1747649, -0.2213654, 0.1617010
4: -0.0908781, 0.1427428, -0.1363116, 0.2151742, -0.3060522, 0.2790544

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
time: 0.30 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
time: 0.32 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0390887, 0.0634215, -0.0993192, 0.0912254
1: -0.0510239, 0.1157805, -0.0557757, 0.1414140, -0.1924379, 0.1715562
2: -0.1029525, 0.1605961, -0.1134923, 0.1820314, -0.2849840, 0.2740884
3: -0.0576805, 0.1386453, -0.0636387, 0.1747649, -0.2324454, 0.2022839
4: -0.1189080, 0.1891569, -0.1363116, 0.2151742, -0.3340822, 0.3254685

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
time: 0.31 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781
time: 0.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.46 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.46
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.46
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.46
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.46
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0303923, 0.0316036, -0.0619959, 0.0619959
1: -0.0419247, 0.0834490, -0.0419247, 0.0834490, -0.1253737, 0.1253737
2: -0.0824195, 0.1223470, -0.0824195, 0.1223470, -0.2047665, 0.2047665
3: -0.0466005, 0.0980623, -0.0466005, 0.0980623, -0.1446628, 0.1446628
4: -0.0908781, 0.1427428, -0.0908781, 0.1427428, -0.2336209, 0.2336209

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880516, upper bound: 0.0894788
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
time: 0.32 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0358978, 0.0521367, -0.0825290, 0.0675014
1: -0.0419247, 0.0834490, -0.0510239, 0.1157805, -0.1577052, 0.1344729
2: -0.0824195, 0.1223470, -0.1029525, 0.1605961, -0.2430156, 0.2252996
3: -0.0466005, 0.0980623, -0.0576805, 0.1386453, -0.1852457, 0.1557429
4: -0.0908781, 0.1427428, -0.1189080, 0.1891569, -0.2800349, 0.2616507

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880516, upper bound: 0.0894834
time: 0.37 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
time: 0.32 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0303923, 0.0316036, -0.0675014, 0.0825290
1: -0.0510239, 0.1157805, -0.0419247, 0.0834490, -0.1344729, 0.1577052
2: -0.1029525, 0.1605961, -0.0824195, 0.1223470, -0.2252996, 0.2430156
3: -0.0576805, 0.1386453, -0.0466005, 0.0980623, -0.1557429, 0.1852458
4: -0.1189080, 0.1891569, -0.0908781, 0.1427428, -0.2616507, 0.2800350

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
time: 0.34 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0358978, 0.0521367, -0.0880345, 0.0880345
1: -0.0510239, 0.1157805, -0.0510239, 0.1157805, -0.1668044, 0.1668043
2: -0.1029525, 0.1605961, -0.1029525, 0.1605961, -0.2635486, 0.2635487
3: -0.0576805, 0.1386453, -0.0576805, 0.1386453, -0.1963258, 0.1963258
4: -0.1189080, 0.1891569, -0.1189080, 0.1891569, -0.3080648, 0.3080649

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317
time: 0.34 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.78 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 0, lower bound: -0.0880516, upper bound: 0.0894788
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.78
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 0, lower bound: -0.0880516, upper bound: 0.0894834
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.78
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.78
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.78
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.78
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0303923, 0.0316036, -0.0601504, 0.0592287
1: -0.0394259, 0.0744049, -0.0419247, 0.0834490, -0.1228749, 0.1163296
2: -0.0761525, 0.1130898, -0.0824195, 0.1223470, -0.1984996, 0.1955093
3: -0.0434286, 0.0854480, -0.0466005, 0.0980623, -0.1414909, 0.1320485
4: -0.0817791, 0.1311970, -0.0908781, 0.1427428, -0.2245218, 0.2220750

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0358978, 0.0521367, -0.0806835, 0.0647341
1: -0.0394259, 0.0744049, -0.0510239, 0.1157805, -0.1552063, 0.1254288
2: -0.0761525, 0.1130898, -0.1029525, 0.1605961, -0.2367487, 0.2160423
3: -0.0434286, 0.0854480, -0.0576805, 0.1386453, -0.1820739, 0.1431285
4: -0.0817791, 0.1311970, -0.1189080, 0.1891569, -0.2709360, 0.2501049

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0303923, 0.0316036, -0.0634249, 0.0704445
1: -0.0462006, 0.0969325, -0.0419247, 0.0834490, -0.1296496, 0.1388572
2: -0.0906157, 0.1402107, -0.0824195, 0.1223470, -0.2129627, 0.2226302
3: -0.0531606, 0.1144974, -0.0466005, 0.0980623, -0.1512229, 0.1610979
4: -0.1036385, 0.1661279, -0.0908781, 0.1427428, -0.2463813, 0.2570060

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0358978, 0.0521367, -0.0839581, 0.0759500
1: -0.0462006, 0.0969325, -0.0510239, 0.1157805, -0.1619810, 0.1479563
2: -0.0906157, 0.1402107, -0.1029525, 0.1605961, -0.2512118, 0.2431632
3: -0.0531606, 0.1144974, -0.0576805, 0.1386453, -0.1918058, 0.1721780
4: -0.1036385, 0.1661279, -0.1189080, 0.1891569, -0.2927953, 0.2850358

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.36 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.76 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
Binary search (step 3): status=Status.VERIFIED, low=0.1875000, high=0.2000000, mid=0.1875000, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 4) starts
Candidate diff: 0.1937500


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895596, upper bound: 0.0897790
time: 0.34 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781
time: 0.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.83 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 0, lower bound: -0.0895596, upper bound: 0.0897790
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0390887, 0.0634215, -0.0938137, 0.0706923
1: -0.0419247, 0.0834490, -0.0557757, 0.1414140, -0.1833387, 0.1392248
2: -0.0824195, 0.1223470, -0.1134923, 0.1820314, -0.2644509, 0.2358393
3: -0.0466005, 0.0980623, -0.0636387, 0.1747649, -0.2213654, 0.1617010
4: -0.0908781, 0.1427428, -0.1363116, 0.2151742, -0.3060522, 0.2790544

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
time: 0.31 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
time: 0.33 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0390887, 0.0634215, -0.0993192, 0.0912254
1: -0.0510239, 0.1157805, -0.0557757, 0.1414140, -0.1924379, 0.1715562
2: -0.1029525, 0.1605961, -0.1134923, 0.1820314, -0.2849840, 0.2740884
3: -0.0576805, 0.1386453, -0.0636387, 0.1747649, -0.2324454, 0.2022839
4: -0.1189080, 0.1891569, -0.1363116, 0.2151742, -0.3340822, 0.3254685

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
time: 0.32 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781
time: 0.31 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.59 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.59
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0303923, 0.0316036, -0.0619959, 0.0619959
1: -0.0419247, 0.0834490, -0.0419247, 0.0834490, -0.1253737, 0.1253737
2: -0.0824195, 0.1223470, -0.0824195, 0.1223470, -0.2047665, 0.2047665
3: -0.0466005, 0.0980623, -0.0466005, 0.0980623, -0.1446628, 0.1446628
4: -0.0908781, 0.1427428, -0.0908781, 0.1427428, -0.2336209, 0.2336209

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880528, upper bound: 0.0894954
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
time: 0.32 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0358978, 0.0521367, -0.0825290, 0.0675014
1: -0.0419247, 0.0834490, -0.0510239, 0.1157805, -0.1577052, 0.1344729
2: -0.0824195, 0.1223470, -0.1029525, 0.1605961, -0.2430156, 0.2252996
3: -0.0466005, 0.0980623, -0.0576805, 0.1386453, -0.1852457, 0.1557429
4: -0.0908781, 0.1427428, -0.1189080, 0.1891569, -0.2800349, 0.2616507

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880528, upper bound: 0.0895118
time: 0.36 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
time: 0.32 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0303923, 0.0316036, -0.0675014, 0.0825290
1: -0.0510239, 0.1157805, -0.0419247, 0.0834490, -0.1344729, 0.1577052
2: -0.1029525, 0.1605961, -0.0824195, 0.1223470, -0.2252996, 0.2430156
3: -0.0576805, 0.1386453, -0.0466005, 0.0980623, -0.1557429, 0.1852458
4: -0.1189080, 0.1891569, -0.0908781, 0.1427428, -0.2616507, 0.2800350

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
time: 0.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0358978, 0.0521367, -0.0880345, 0.0880345
1: -0.0510239, 0.1157805, -0.0510239, 0.1157805, -0.1668044, 0.1668043
2: -0.1029525, 0.1605961, -0.1029525, 0.1605961, -0.2635486, 0.2635487
3: -0.0576805, 0.1386453, -0.0576805, 0.1386453, -0.1963258, 0.1963258
4: -0.1189080, 0.1891569, -0.1189080, 0.1891569, -0.3080648, 0.3080649

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
time: 0.32 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317
time: 0.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.37 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.37
Output dim: 0, lower bound: -0.0880528, upper bound: 0.0894954
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.37
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.37
Output dim: 0, lower bound: -0.0880528, upper bound: 0.0895118
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.37
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.37
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.37
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.37
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.37
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0303923, 0.0316036, -0.0601504, 0.0592287
1: -0.0394259, 0.0744049, -0.0419247, 0.0834490, -0.1228749, 0.1163296
2: -0.0761525, 0.1130898, -0.0824195, 0.1223470, -0.1984996, 0.1955093
3: -0.0434286, 0.0854480, -0.0466005, 0.0980623, -0.1414909, 0.1320485
4: -0.0817791, 0.1311970, -0.0908781, 0.1427428, -0.2245218, 0.2220750

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.32 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0358978, 0.0521367, -0.0806835, 0.0647341
1: -0.0394259, 0.0744049, -0.0510239, 0.1157805, -0.1552063, 0.1254288
2: -0.0761525, 0.1130898, -0.1029525, 0.1605961, -0.2367487, 0.2160423
3: -0.0434286, 0.0854480, -0.0576805, 0.1386453, -0.1820739, 0.1431285
4: -0.0817791, 0.1311970, -0.1189080, 0.1891569, -0.2709360, 0.2501049

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0303923, 0.0316036, -0.0634249, 0.0704445
1: -0.0462006, 0.0969325, -0.0419247, 0.0834490, -0.1296496, 0.1388572
2: -0.0906157, 0.1402107, -0.0824195, 0.1223470, -0.2129627, 0.2226302
3: -0.0531606, 0.1144974, -0.0466005, 0.0980623, -0.1512229, 0.1610979
4: -0.1036385, 0.1661279, -0.0908781, 0.1427428, -0.2463813, 0.2570060

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0358978, 0.0521367, -0.0839581, 0.0759500
1: -0.0462006, 0.0969325, -0.0510239, 0.1157805, -0.1619810, 0.1479563
2: -0.0906157, 0.1402107, -0.1029525, 0.1605961, -0.2512118, 0.2431632
3: -0.0531606, 0.1144974, -0.0576805, 0.1386453, -0.1918058, 0.1721780
4: -0.1036385, 0.1661279, -0.1189080, 0.1891569, -0.2927953, 0.2850358

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.37 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.37
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.37
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.37
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.37
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.37
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.37
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.37
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.37
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
Binary search (step 4): status=Status.VERIFIED, low=0.1937500, high=0.2000000, mid=0.1937500, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 5) starts
Candidate diff: 0.1968750


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895599, upper bound: 0.0897790
time: 0.33 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781
time: 0.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.79 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -0.0895599, upper bound: 0.0897790
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0390887, 0.0634215, -0.0938137, 0.0706923
1: -0.0419247, 0.0834490, -0.0557757, 0.1414140, -0.1833387, 0.1392248
2: -0.0824195, 0.1223470, -0.1134923, 0.1820314, -0.2644509, 0.2358393
3: -0.0466005, 0.0980623, -0.0636387, 0.1747649, -0.2213654, 0.1617010
4: -0.0908781, 0.1427428, -0.1363116, 0.2151742, -0.3060522, 0.2790544

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
time: 0.29 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
time: 0.32 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0390887, 0.0634215, -0.0993192, 0.0912254
1: -0.0510239, 0.1157805, -0.0557757, 0.1414140, -0.1924379, 0.1715562
2: -0.1029525, 0.1605961, -0.1134923, 0.1820314, -0.2849840, 0.2740884
3: -0.0576805, 0.1386453, -0.0636387, 0.1747649, -0.2324454, 0.2022839
4: -0.1189080, 0.1891569, -0.1363116, 0.2151742, -0.3340822, 0.3254685

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
time: 0.29 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781
time: 0.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.20 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.20
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0303923, 0.0316036, -0.0619959, 0.0619959
1: -0.0419247, 0.0834490, -0.0419247, 0.0834490, -0.1253737, 0.1253737
2: -0.0824195, 0.1223470, -0.0824195, 0.1223470, -0.2047665, 0.2047665
3: -0.0466005, 0.0980623, -0.0466005, 0.0980623, -0.1446628, 0.1446628
4: -0.0908781, 0.1427428, -0.0908781, 0.1427428, -0.2336209, 0.2336209

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
time: 0.30 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0358978, 0.0521367, -0.0825290, 0.0675014
1: -0.0419247, 0.0834490, -0.0510239, 0.1157805, -0.1577052, 0.1344729
2: -0.0824195, 0.1223470, -0.1029525, 0.1605961, -0.2430156, 0.2252996
3: -0.0466005, 0.0980623, -0.0576805, 0.1386453, -0.1852457, 0.1557429
4: -0.0908781, 0.1427428, -0.1189080, 0.1891569, -0.2800349, 0.2616507

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0895260
time: 0.30 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
time: 0.30 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0303923, 0.0316036, -0.0675014, 0.0825290
1: -0.0510239, 0.1157805, -0.0419247, 0.0834490, -0.1344729, 0.1577052
2: -0.1029525, 0.1605961, -0.0824195, 0.1223470, -0.2252996, 0.2430156
3: -0.0576805, 0.1386453, -0.0466005, 0.0980623, -0.1557429, 0.1852458
4: -0.1189080, 0.1891569, -0.0908781, 0.1427428, -0.2616507, 0.2800350

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
time: 0.30 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
time: 0.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0358978, 0.0521367, -0.0880345, 0.0880345
1: -0.0510239, 0.1157805, -0.0510239, 0.1157805, -0.1668044, 0.1668043
2: -0.1029525, 0.1605961, -0.1029525, 0.1605961, -0.2635486, 0.2635487
3: -0.0576805, 0.1386453, -0.0576805, 0.1386453, -0.1963258, 0.1963258
4: -0.1189080, 0.1891569, -0.1189080, 0.1891569, -0.3080648, 0.3080649

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317
time: 0.33 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.30 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.30
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0895260
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.30
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.30
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.30
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.30
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0303923, 0.0316036, -0.0601504, 0.0592287
1: -0.0394259, 0.0744049, -0.0419247, 0.0834490, -0.1228749, 0.1163296
2: -0.0761525, 0.1130898, -0.0824195, 0.1223470, -0.1984996, 0.1955093
3: -0.0434286, 0.0854480, -0.0466005, 0.0980623, -0.1414909, 0.1320485
4: -0.0817791, 0.1311970, -0.0908781, 0.1427428, -0.2245218, 0.2220750

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0358978, 0.0521367, -0.0806835, 0.0647341
1: -0.0394259, 0.0744049, -0.0510239, 0.1157805, -0.1552063, 0.1254288
2: -0.0761525, 0.1130898, -0.1029525, 0.1605961, -0.2367487, 0.2160423
3: -0.0434286, 0.0854480, -0.0576805, 0.1386453, -0.1820739, 0.1431285
4: -0.0817791, 0.1311970, -0.1189080, 0.1891569, -0.2709360, 0.2501049

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.35 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0303923, 0.0316036, -0.0634249, 0.0704445
1: -0.0462006, 0.0969325, -0.0419247, 0.0834490, -0.1296496, 0.1388572
2: -0.0906157, 0.1402107, -0.0824195, 0.1223470, -0.2129627, 0.2226302
3: -0.0531606, 0.1144974, -0.0466005, 0.0980623, -0.1512229, 0.1610979
4: -0.1036385, 0.1661279, -0.0908781, 0.1427428, -0.2463813, 0.2570060

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0358978, 0.0521367, -0.0839581, 0.0759500
1: -0.0462006, 0.0969325, -0.0510239, 0.1157805, -0.1619810, 0.1479563
2: -0.0906157, 0.1402107, -0.1029525, 0.1605961, -0.2512118, 0.2431632
3: -0.0531606, 0.1144974, -0.0576805, 0.1386453, -0.1918058, 0.1721780
4: -0.1036385, 0.1661279, -0.1189080, 0.1891569, -0.2927953, 0.2850358

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.35 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.64 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.64
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
Binary search (step 5): status=Status.VERIFIED, low=0.1968750, high=0.2000000, mid=0.1968750, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 6) starts
Candidate diff: 0.1984375


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895600, upper bound: 0.0897790
time: 0.34 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781
time: 0.33 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.83 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 0, lower bound: -0.0895600, upper bound: 0.0897790
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0390887, 0.0634215, -0.0938137, 0.0706923
1: -0.0419247, 0.0834490, -0.0557757, 0.1414140, -0.1833387, 0.1392248
2: -0.0824195, 0.1223470, -0.1134923, 0.1820314, -0.2644509, 0.2358393
3: -0.0466005, 0.0980623, -0.0636387, 0.1747649, -0.2213654, 0.1617010
4: -0.0908781, 0.1427428, -0.1363116, 0.2151742, -0.3060522, 0.2790544

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
time: 0.31 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
time: 0.33 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0390887, 0.0634215, -0.0993192, 0.0912254
1: -0.0510239, 0.1157805, -0.0557757, 0.1414140, -0.1924379, 0.1715562
2: -0.1029525, 0.1605961, -0.1134923, 0.1820314, -0.2849840, 0.2740884
3: -0.0576805, 0.1386453, -0.0636387, 0.1747649, -0.2324454, 0.2022839
4: -0.1189080, 0.1891569, -0.1363116, 0.2151742, -0.3340822, 0.3254685

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
time: 0.30 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781
time: 0.31 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.48 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.48
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.48
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.48
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.48
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0303923, 0.0316036, -0.0619959, 0.0619959
1: -0.0419247, 0.0834490, -0.0419247, 0.0834490, -0.1253737, 0.1253737
2: -0.0824195, 0.1223470, -0.0824195, 0.1223470, -0.2047665, 0.2047665
3: -0.0466005, 0.0980623, -0.0466005, 0.0980623, -0.1446628, 0.1446628
4: -0.0908781, 0.1427428, -0.0908781, 0.1427428, -0.2336209, 0.2336209

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
time: 0.31 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0358978, 0.0521367, -0.0825290, 0.0675014
1: -0.0419247, 0.0834490, -0.0510239, 0.1157805, -0.1577052, 0.1344729
2: -0.0824195, 0.1223470, -0.1029525, 0.1605961, -0.2430156, 0.2252996
3: -0.0466005, 0.0980623, -0.0576805, 0.1386453, -0.1852457, 0.1557429
4: -0.0908781, 0.1427428, -0.1189080, 0.1891569, -0.2800349, 0.2616507

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
time: 0.31 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0303923, 0.0316036, -0.0675014, 0.0825290
1: -0.0510239, 0.1157805, -0.0419247, 0.0834490, -0.1344729, 0.1577052
2: -0.1029525, 0.1605961, -0.0824195, 0.1223470, -0.2252996, 0.2430156
3: -0.0576805, 0.1386453, -0.0466005, 0.0980623, -0.1557429, 0.1852458
4: -0.1189080, 0.1891569, -0.0908781, 0.1427428, -0.2616507, 0.2800350

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
time: 0.32 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0358978, 0.0521367, -0.0880345, 0.0880345
1: -0.0510239, 0.1157805, -0.0510239, 0.1157805, -0.1668044, 0.1668043
2: -0.1029525, 0.1605961, -0.1029525, 0.1605961, -0.2635486, 0.2635487
3: -0.0576805, 0.1386453, -0.0576805, 0.1386453, -0.1963258, 0.1963258
4: -0.1189080, 0.1891569, -0.1189080, 0.1891569, -0.3080648, 0.3080649

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317
time: 0.32 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.52 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0303923, 0.0316036, -0.0601504, 0.0592287
1: -0.0394259, 0.0744049, -0.0419247, 0.0834490, -0.1228749, 0.1163296
2: -0.0761525, 0.1130898, -0.0824195, 0.1223470, -0.1984996, 0.1955093
3: -0.0434286, 0.0854480, -0.0466005, 0.0980623, -0.1414909, 0.1320485
4: -0.0817791, 0.1311970, -0.0908781, 0.1427428, -0.2245218, 0.2220750

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0358978, 0.0521367, -0.0806835, 0.0647341
1: -0.0394259, 0.0744049, -0.0510239, 0.1157805, -0.1552063, 0.1254288
2: -0.0761525, 0.1130898, -0.1029525, 0.1605961, -0.2367487, 0.2160423
3: -0.0434286, 0.0854480, -0.0576805, 0.1386453, -0.1820739, 0.1431285
4: -0.0817791, 0.1311970, -0.1189080, 0.1891569, -0.2709360, 0.2501049

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879592
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879592
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0303923, 0.0316036, -0.0634249, 0.0704445
1: -0.0462006, 0.0969325, -0.0419247, 0.0834490, -0.1296496, 0.1388572
2: -0.0906157, 0.1402107, -0.0824195, 0.1223470, -0.2129627, 0.2226302
3: -0.0531606, 0.1144974, -0.0466005, 0.0980623, -0.1512229, 0.1610979
4: -0.1036385, 0.1661279, -0.0908781, 0.1427428, -0.2463813, 0.2570060

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0358978, 0.0521367, -0.0839581, 0.0759500
1: -0.0462006, 0.0969325, -0.0510239, 0.1157805, -0.1619810, 0.1479563
2: -0.0906157, 0.1402107, -0.1029525, 0.1605961, -0.2512118, 0.2431632
3: -0.0531606, 0.1144974, -0.0576805, 0.1386453, -0.1918058, 0.1721780
4: -0.1036385, 0.1661279, -0.1189080, 0.1891569, -0.2927953, 0.2850358

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.38 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879592
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879592
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.38
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
Binary search (step 6): status=Status.VERIFIED, low=0.1984375, high=0.2000000, mid=0.1984375, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 7) starts
Candidate diff: 0.1992187


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895600, upper bound: 0.0897790
time: 0.33 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781
time: 0.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.79 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -0.0895600, upper bound: 0.0897790
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0390887, 0.0634215, -0.0938137, 0.0706923
1: -0.0419247, 0.0834490, -0.0557757, 0.1414140, -0.1833387, 0.1392248
2: -0.0824195, 0.1223470, -0.1134923, 0.1820314, -0.2644509, 0.2358393
3: -0.0466005, 0.0980623, -0.0636387, 0.1747649, -0.2213654, 0.1617010
4: -0.0908781, 0.1427428, -0.1363116, 0.2151742, -0.3060522, 0.2790544

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
time: 0.30 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
time: 0.32 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0390887, 0.0634215, -0.0993192, 0.0912254
1: -0.0510239, 0.1157805, -0.0557757, 0.1414140, -0.1924379, 0.1715562
2: -0.1029525, 0.1605961, -0.1134923, 0.1820314, -0.2849840, 0.2740884
3: -0.0576805, 0.1386453, -0.0636387, 0.1747649, -0.2324454, 0.2022839
4: -0.1189080, 0.1891569, -0.1363116, 0.2151742, -0.3340822, 0.3254685

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
time: 0.31 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781
time: 0.33 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.30 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.30
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0303923, 0.0316036, -0.0619959, 0.0619959
1: -0.0419247, 0.0834490, -0.0419247, 0.0834490, -0.1253737, 0.1253737
2: -0.0824195, 0.1223470, -0.0824195, 0.1223470, -0.2047665, 0.2047665
3: -0.0466005, 0.0980623, -0.0466005, 0.0980623, -0.1446628, 0.1446628
4: -0.0908781, 0.1427428, -0.0908781, 0.1427428, -0.2336209, 0.2336209

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
time: 0.31 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0358978, 0.0521367, -0.0825290, 0.0675014
1: -0.0419247, 0.0834490, -0.0510239, 0.1157805, -0.1577052, 0.1344729
2: -0.0824195, 0.1223470, -0.1029525, 0.1605961, -0.2430156, 0.2252996
3: -0.0466005, 0.0980623, -0.0576805, 0.1386453, -0.1852457, 0.1557429
4: -0.0908781, 0.1427428, -0.1189080, 0.1891569, -0.2800349, 0.2616507

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
time: 0.30 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0303923, 0.0316036, -0.0675014, 0.0825290
1: -0.0510239, 0.1157805, -0.0419247, 0.0834490, -0.1344729, 0.1577052
2: -0.1029525, 0.1605961, -0.0824195, 0.1223470, -0.2252996, 0.2430156
3: -0.0576805, 0.1386453, -0.0466005, 0.0980623, -0.1557429, 0.1852458
4: -0.1189080, 0.1891569, -0.0908781, 0.1427428, -0.2616507, 0.2800350

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
time: 0.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0358978, 0.0521367, -0.0880345, 0.0880345
1: -0.0510239, 0.1157805, -0.0510239, 0.1157805, -0.1668044, 0.1668043
2: -0.1029525, 0.1605961, -0.1029525, 0.1605961, -0.2635486, 0.2635487
3: -0.0576805, 0.1386453, -0.0576805, 0.1386453, -0.1963258, 0.1963258
4: -0.1189080, 0.1891569, -0.1189080, 0.1891569, -0.3080648, 0.3080649

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317
time: 0.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.32 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.32
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0303923, 0.0316036, -0.0601504, 0.0592287
1: -0.0394259, 0.0744049, -0.0419247, 0.0834490, -0.1228749, 0.1163296
2: -0.0761525, 0.1130898, -0.0824195, 0.1223470, -0.1984996, 0.1955093
3: -0.0434286, 0.0854480, -0.0466005, 0.0980623, -0.1414909, 0.1320485
4: -0.0817791, 0.1311970, -0.0908781, 0.1427428, -0.2245218, 0.2220750

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.32 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0358978, 0.0521367, -0.0806835, 0.0647341
1: -0.0394259, 0.0744049, -0.0510239, 0.1157805, -0.1552063, 0.1254288
2: -0.0761525, 0.1130898, -0.1029525, 0.1605961, -0.2367487, 0.2160423
3: -0.0434286, 0.0854480, -0.0576805, 0.1386453, -0.1820739, 0.1431285
4: -0.0817791, 0.1311970, -0.1189080, 0.1891569, -0.2709360, 0.2501049

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0303923, 0.0316036, -0.0634249, 0.0704445
1: -0.0462006, 0.0969325, -0.0419247, 0.0834490, -0.1296496, 0.1388572
2: -0.0906157, 0.1402107, -0.0824195, 0.1223470, -0.2129627, 0.2226302
3: -0.0531606, 0.1144974, -0.0466005, 0.0980623, -0.1512229, 0.1610979
4: -0.1036385, 0.1661279, -0.0908781, 0.1427428, -0.2463813, 0.2570060

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0358978, 0.0521367, -0.0839581, 0.0759500
1: -0.0462006, 0.0969325, -0.0510239, 0.1157805, -0.1619810, 0.1479563
2: -0.0906157, 0.1402107, -0.1029525, 0.1605961, -0.2512118, 0.2431632
3: -0.0531606, 0.1144974, -0.0576805, 0.1386453, -0.1918058, 0.1721780
4: -0.1036385, 0.1661279, -0.1189080, 0.1891569, -0.2927953, 0.2850358

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.35 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.66 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
Binary search (step 7): status=Status.VERIFIED, low=0.1992187, high=0.2000000, mid=0.1992187, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 8) starts
Candidate diff: 0.1996094


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895600, upper bound: 0.0897790
time: 0.35 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895271, upper bound: 0.0895781
time: 0.33 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.83 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 0, lower bound: -0.0895600, upper bound: 0.0897790
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 0, lower bound: -0.0895271, upper bound: 0.0895781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0390887, 0.0634215, -0.0938137, 0.0706923
1: -0.0419247, 0.0834490, -0.0557757, 0.1414140, -0.1833387, 0.1392248
2: -0.0824195, 0.1223470, -0.1134923, 0.1820314, -0.2644509, 0.2358393
3: -0.0466005, 0.0980623, -0.0636387, 0.1747649, -0.2213654, 0.1617010
4: -0.0908781, 0.1427428, -0.1363116, 0.2151742, -0.3060522, 0.2790544

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
time: 0.46 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
time: 0.33 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0390887, 0.0634215, -0.0993192, 0.0912254
1: -0.0510239, 0.1157805, -0.0557757, 0.1414140, -0.1924379, 0.1715562
2: -0.1029525, 0.1605961, -0.1134923, 0.1820314, -0.2849840, 0.2740884
3: -0.0576805, 0.1386453, -0.0636387, 0.1747649, -0.2324454, 0.2022839
4: -0.1189080, 0.1891569, -0.1363116, 0.2151742, -0.3340822, 0.3254685

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
time: 0.31 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781
time: 0.31 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.57 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.57
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0303923, 0.0316036, -0.0619959, 0.0619959
1: -0.0419247, 0.0834490, -0.0419247, 0.0834490, -0.1253737, 0.1253737
2: -0.0824195, 0.1223470, -0.0824195, 0.1223470, -0.2047665, 0.2047665
3: -0.0466005, 0.0980623, -0.0466005, 0.0980623, -0.1446628, 0.1446628
4: -0.0908781, 0.1427428, -0.0908781, 0.1427428, -0.2336209, 0.2336209

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
time: 0.31 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0358978, 0.0521367, -0.0825290, 0.0675014
1: -0.0419247, 0.0834490, -0.0510239, 0.1157805, -0.1577052, 0.1344729
2: -0.0824195, 0.1223470, -0.1029525, 0.1605961, -0.2430156, 0.2252996
3: -0.0466005, 0.0980623, -0.0576805, 0.1386453, -0.1852457, 0.1557429
4: -0.0908781, 0.1427428, -0.1189080, 0.1891569, -0.2800349, 0.2616507

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
time: 0.31 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0303923, 0.0316036, -0.0675014, 0.0825290
1: -0.0510239, 0.1157805, -0.0419247, 0.0834490, -0.1344729, 0.1577052
2: -0.1029525, 0.1605961, -0.0824195, 0.1223470, -0.2252996, 0.2430156
3: -0.0576805, 0.1386453, -0.0466005, 0.0980623, -0.1557429, 0.1852458
4: -0.1189080, 0.1891569, -0.0908781, 0.1427428, -0.2616507, 0.2800350

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
time: 0.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0358978, 0.0521367, -0.0880345, 0.0880345
1: -0.0510239, 0.1157805, -0.0510239, 0.1157805, -0.1668044, 0.1668043
2: -0.1029525, 0.1605961, -0.1029525, 0.1605961, -0.2635486, 0.2635487
3: -0.0576805, 0.1386453, -0.0576805, 0.1386453, -0.1963258, 0.1963258
4: -0.1189080, 0.1891569, -0.1189080, 0.1891569, -0.3080648, 0.3080649

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317
time: 0.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.53 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0303923, 0.0316036, -0.0601504, 0.0592287
1: -0.0394259, 0.0744049, -0.0419247, 0.0834490, -0.1228749, 0.1163296
2: -0.0761525, 0.1130898, -0.0824195, 0.1223470, -0.1984996, 0.1955093
3: -0.0434286, 0.0854480, -0.0466005, 0.0980623, -0.1414909, 0.1320485
4: -0.0817791, 0.1311970, -0.0908781, 0.1427428, -0.2245218, 0.2220750

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0358978, 0.0521367, -0.0806835, 0.0647341
1: -0.0394259, 0.0744049, -0.0510239, 0.1157805, -0.1552063, 0.1254288
2: -0.0761525, 0.1130898, -0.1029525, 0.1605961, -0.2367487, 0.2160423
3: -0.0434286, 0.0854480, -0.0576805, 0.1386453, -0.1820739, 0.1431285
4: -0.0817791, 0.1311970, -0.1189080, 0.1891569, -0.2709360, 0.2501049

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879592
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879592
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0303923, 0.0316036, -0.0634249, 0.0704445
1: -0.0462006, 0.0969325, -0.0419247, 0.0834490, -0.1296496, 0.1388572
2: -0.0906157, 0.1402107, -0.0824195, 0.1223470, -0.2129627, 0.2226302
3: -0.0531606, 0.1144974, -0.0466005, 0.0980623, -0.1512229, 0.1610979
4: -0.1036385, 0.1661279, -0.0908781, 0.1427428, -0.2463813, 0.2570060

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0358978, 0.0521367, -0.0839581, 0.0759500
1: -0.0462006, 0.0969325, -0.0510239, 0.1157805, -0.1619810, 0.1479563
2: -0.0906157, 0.1402107, -0.1029525, 0.1605961, -0.2512118, 0.2431632
3: -0.0531606, 0.1144974, -0.0576805, 0.1386453, -0.1918058, 0.1721780
4: -0.1036385, 0.1661279, -0.1189080, 0.1891569, -0.2927953, 0.2850358

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.44 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879592
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879592
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.44
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
Binary search (step 8): status=Status.VERIFIED, low=0.1996094, high=0.2000000, mid=0.1996094, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 9) starts
Candidate diff: 0.1998047


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895600, upper bound: 0.0897790
time: 0.33 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781
time: 0.31 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.79 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -0.0895600, upper bound: 0.0897790
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0390887, 0.0634215, -0.0938137, 0.0706923
1: -0.0419247, 0.0834490, -0.0557757, 0.1414140, -0.1833387, 0.1392248
2: -0.0824195, 0.1223470, -0.1134923, 0.1820314, -0.2644509, 0.2358393
3: -0.0466005, 0.0980623, -0.0636387, 0.1747649, -0.2213654, 0.1617010
4: -0.0908781, 0.1427428, -0.1363116, 0.2151742, -0.3060522, 0.2790544

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
time: 0.31 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
time: 0.30 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0390887, 0.0634215, -0.0993192, 0.0912254
1: -0.0510239, 0.1157805, -0.0557757, 0.1414140, -0.1924379, 0.1715562
2: -0.1029525, 0.1605961, -0.1134923, 0.1820314, -0.2849840, 0.2740884
3: -0.0576805, 0.1386453, -0.0636387, 0.1747649, -0.2324454, 0.2022839
4: -0.1189080, 0.1891569, -0.1363116, 0.2151742, -0.3340822, 0.3254685

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
time: 0.30 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781
time: 0.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.29 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.29
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0303923, 0.0316036, -0.0619959, 0.0619959
1: -0.0419247, 0.0834490, -0.0419247, 0.0834490, -0.1253737, 0.1253737
2: -0.0824195, 0.1223470, -0.0824195, 0.1223470, -0.2047665, 0.2047665
3: -0.0466005, 0.0980623, -0.0466005, 0.0980623, -0.1446628, 0.1446628
4: -0.0908781, 0.1427428, -0.0908781, 0.1427428, -0.2336209, 0.2336209

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
time: 0.31 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0358978, 0.0521367, -0.0825290, 0.0675014
1: -0.0419247, 0.0834490, -0.0510239, 0.1157805, -0.1577052, 0.1344729
2: -0.0824195, 0.1223470, -0.1029525, 0.1605961, -0.2430156, 0.2252996
3: -0.0466005, 0.0980623, -0.0576805, 0.1386453, -0.1852457, 0.1557429
4: -0.0908781, 0.1427428, -0.1189080, 0.1891569, -0.2800349, 0.2616507

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
time: 0.30 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
time: 0.31 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0303923, 0.0316036, -0.0675014, 0.0825290
1: -0.0510239, 0.1157805, -0.0419247, 0.0834490, -0.1344729, 0.1577052
2: -0.1029525, 0.1605961, -0.0824195, 0.1223470, -0.2252996, 0.2430156
3: -0.0576805, 0.1386453, -0.0466005, 0.0980623, -0.1557429, 0.1852458
4: -0.1189080, 0.1891569, -0.0908781, 0.1427428, -0.2616507, 0.2800350

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
time: 0.32 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0358978, 0.0521367, -0.0880345, 0.0880345
1: -0.0510239, 0.1157805, -0.0510239, 0.1157805, -0.1668044, 0.1668043
2: -0.1029525, 0.1605961, -0.1029525, 0.1605961, -0.2635486, 0.2635487
3: -0.0576805, 0.1386453, -0.0576805, 0.1386453, -0.1963258, 0.1963258
4: -0.1189080, 0.1891569, -0.1189080, 0.1891569, -0.3080648, 0.3080649

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317
time: 0.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.28 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0303923, 0.0316036, -0.0601504, 0.0592287
1: -0.0394259, 0.0744049, -0.0419247, 0.0834490, -0.1228749, 0.1163296
2: -0.0761525, 0.1130898, -0.0824195, 0.1223470, -0.1984996, 0.1955093
3: -0.0434286, 0.0854480, -0.0466005, 0.0980623, -0.1414909, 0.1320485
4: -0.0817791, 0.1311970, -0.0908781, 0.1427428, -0.2245218, 0.2220750

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.32 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0358978, 0.0521367, -0.0806835, 0.0647341
1: -0.0394259, 0.0744049, -0.0510239, 0.1157805, -0.1552063, 0.1254288
2: -0.0761525, 0.1130898, -0.1029525, 0.1605961, -0.2367487, 0.2160423
3: -0.0434286, 0.0854480, -0.0576805, 0.1386453, -0.1820739, 0.1431285
4: -0.0817791, 0.1311970, -0.1189080, 0.1891569, -0.2709360, 0.2501049

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0303923, 0.0316036, -0.0634249, 0.0704445
1: -0.0462006, 0.0969325, -0.0419247, 0.0834490, -0.1296496, 0.1388572
2: -0.0906157, 0.1402107, -0.0824195, 0.1223470, -0.2129627, 0.2226302
3: -0.0531606, 0.1144974, -0.0466005, 0.0980623, -0.1512229, 0.1610979
4: -0.1036385, 0.1661279, -0.0908781, 0.1427428, -0.2463813, 0.2570060

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0358978, 0.0521367, -0.0839581, 0.0759500
1: -0.0462006, 0.0969325, -0.0510239, 0.1157805, -0.1619810, 0.1479563
2: -0.0906157, 0.1402107, -0.1029525, 0.1605961, -0.2512118, 0.2431632
3: -0.0531606, 0.1144974, -0.0576805, 0.1386453, -0.1918058, 0.1721780
4: -0.1036385, 0.1661279, -0.1189080, 0.1891569, -0.2927953, 0.2850358

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.36 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.54 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.54
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
Binary search (step 9): status=Status.VERIFIED, low=0.1998047, high=0.2000000, mid=0.1998047, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 10) starts
Candidate diff: 0.1999023


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895600, upper bound: 0.0897790
time: 0.34 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781
time: 0.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.83 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 0, lower bound: -0.0895600, upper bound: 0.0897790
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0390887, 0.0634215, -0.0938137, 0.0706923
1: -0.0419247, 0.0834490, -0.0557757, 0.1414140, -0.1833387, 0.1392248
2: -0.0824195, 0.1223470, -0.1134923, 0.1820314, -0.2644509, 0.2358393
3: -0.0466005, 0.0980623, -0.0636387, 0.1747649, -0.2213654, 0.1617010
4: -0.0908781, 0.1427428, -0.1363116, 0.2151742, -0.3060522, 0.2790544

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
time: 0.32 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
time: 0.31 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0390887, 0.0634215, -0.0993192, 0.0912254
1: -0.0510239, 0.1157805, -0.0557757, 0.1414140, -0.1924379, 0.1715562
2: -0.1029525, 0.1605961, -0.1134923, 0.1820314, -0.2849840, 0.2740884
3: -0.0576805, 0.1386453, -0.0636387, 0.1747649, -0.2324454, 0.2022839
4: -0.1189080, 0.1891569, -0.1363116, 0.2151742, -0.3340822, 0.3254685

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
time: 0.30 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781
time: 0.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.48 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.48
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.48
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.48
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.48
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0303923, 0.0316036, -0.0619959, 0.0619959
1: -0.0419247, 0.0834490, -0.0419247, 0.0834490, -0.1253737, 0.1253737
2: -0.0824195, 0.1223470, -0.0824195, 0.1223470, -0.2047665, 0.2047665
3: -0.0466005, 0.0980623, -0.0466005, 0.0980623, -0.1446628, 0.1446628
4: -0.0908781, 0.1427428, -0.0908781, 0.1427428, -0.2336209, 0.2336209

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
time: 0.31 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0358978, 0.0521367, -0.0825290, 0.0675014
1: -0.0419247, 0.0834490, -0.0510239, 0.1157805, -0.1577052, 0.1344729
2: -0.0824195, 0.1223470, -0.1029525, 0.1605961, -0.2430156, 0.2252996
3: -0.0466005, 0.0980623, -0.0576805, 0.1386453, -0.1852457, 0.1557429
4: -0.0908781, 0.1427428, -0.1189080, 0.1891569, -0.2800349, 0.2616507

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
time: 0.31 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0303923, 0.0316036, -0.0675014, 0.0825290
1: -0.0510239, 0.1157805, -0.0419247, 0.0834490, -0.1344729, 0.1577052
2: -0.1029525, 0.1605961, -0.0824195, 0.1223470, -0.2252996, 0.2430156
3: -0.0576805, 0.1386453, -0.0466005, 0.0980623, -0.1557429, 0.1852458
4: -0.1189080, 0.1891569, -0.0908781, 0.1427428, -0.2616507, 0.2800350

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
time: 0.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0358978, 0.0521367, -0.0880345, 0.0880345
1: -0.0510239, 0.1157805, -0.0510239, 0.1157805, -0.1668044, 0.1668043
2: -0.1029525, 0.1605961, -0.1029525, 0.1605961, -0.2635486, 0.2635487
3: -0.0576805, 0.1386453, -0.0576805, 0.1386453, -0.1963258, 0.1963258
4: -0.1189080, 0.1891569, -0.1189080, 0.1891569, -0.3080648, 0.3080649

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317
time: 0.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.56 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.56
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0303923, 0.0316036, -0.0601504, 0.0592287
1: -0.0394259, 0.0744049, -0.0419247, 0.0834490, -0.1228749, 0.1163296
2: -0.0761525, 0.1130898, -0.0824195, 0.1223470, -0.1984996, 0.1955093
3: -0.0434286, 0.0854480, -0.0466005, 0.0980623, -0.1414909, 0.1320485
4: -0.0817791, 0.1311970, -0.0908781, 0.1427428, -0.2245218, 0.2220750

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0358978, 0.0521367, -0.0806835, 0.0647341
1: -0.0394259, 0.0744049, -0.0510239, 0.1157805, -0.1552063, 0.1254288
2: -0.0761525, 0.1130898, -0.1029525, 0.1605961, -0.2367487, 0.2160423
3: -0.0434286, 0.0854480, -0.0576805, 0.1386453, -0.1820739, 0.1431285
4: -0.0817791, 0.1311970, -0.1189080, 0.1891569, -0.2709360, 0.2501049

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0303923, 0.0316036, -0.0634249, 0.0704445
1: -0.0462006, 0.0969325, -0.0419247, 0.0834490, -0.1296496, 0.1388572
2: -0.0906157, 0.1402107, -0.0824195, 0.1223470, -0.2129627, 0.2226302
3: -0.0531606, 0.1144974, -0.0466005, 0.0980623, -0.1512229, 0.1610979
4: -0.1036385, 0.1661279, -0.0908781, 0.1427428, -0.2463813, 0.2570060

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0358978, 0.0521367, -0.0839581, 0.0759500
1: -0.0462006, 0.0969325, -0.0510239, 0.1157805, -0.1619810, 0.1479563
2: -0.0906157, 0.1402107, -0.1029525, 0.1605961, -0.2512118, 0.2431632
3: -0.0531606, 0.1144974, -0.0576805, 0.1386453, -0.1918058, 0.1721780
4: -0.1036385, 0.1661279, -0.1189080, 0.1891569, -0.2927953, 0.2850358

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.35 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.56 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.56
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
Binary search (step 10): status=Status.VERIFIED, low=0.1999023, high=0.2000000, mid=0.1999023, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 11) starts
Candidate diff: 0.1999512


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895600, upper bound: 0.0897790
time: 0.33 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781
time: 0.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.78 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -0.0895600, upper bound: 0.0897790
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0390887, 0.0634215, -0.0938137, 0.0706923
1: -0.0419247, 0.0834490, -0.0557757, 0.1414140, -0.1833387, 0.1392248
2: -0.0824195, 0.1223470, -0.1134923, 0.1820314, -0.2644509, 0.2358393
3: -0.0466005, 0.0980623, -0.0636387, 0.1747649, -0.2213654, 0.1617010
4: -0.0908781, 0.1427428, -0.1363116, 0.2151742, -0.3060522, 0.2790544

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
time: 0.30 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
time: 0.30 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0390887, 0.0634215, -0.0993192, 0.0912254
1: -0.0510239, 0.1157805, -0.0557757, 0.1414140, -0.1924379, 0.1715562
2: -0.1029525, 0.1605961, -0.1134923, 0.1820314, -0.2849840, 0.2740884
3: -0.0576805, 0.1386453, -0.0636387, 0.1747649, -0.2324454, 0.2022839
4: -0.1189080, 0.1891569, -0.1363116, 0.2151742, -0.3340822, 0.3254685

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
time: 0.29 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781
time: 0.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.23 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0303923, 0.0316036, -0.0619959, 0.0619959
1: -0.0419247, 0.0834490, -0.0419247, 0.0834490, -0.1253737, 0.1253737
2: -0.0824195, 0.1223470, -0.0824195, 0.1223470, -0.2047665, 0.2047665
3: -0.0466005, 0.0980623, -0.0466005, 0.0980623, -0.1446628, 0.1446628
4: -0.0908781, 0.1427428, -0.0908781, 0.1427428, -0.2336209, 0.2336209

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
time: 0.30 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0358978, 0.0521367, -0.0825290, 0.0675014
1: -0.0419247, 0.0834490, -0.0510239, 0.1157805, -0.1577052, 0.1344729
2: -0.0824195, 0.1223470, -0.1029525, 0.1605961, -0.2430156, 0.2252996
3: -0.0466005, 0.0980623, -0.0576805, 0.1386453, -0.1852457, 0.1557429
4: -0.0908781, 0.1427428, -0.1189080, 0.1891569, -0.2800349, 0.2616507

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
time: 0.30 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
time: 0.30 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0303923, 0.0316036, -0.0675014, 0.0825290
1: -0.0510239, 0.1157805, -0.0419247, 0.0834490, -0.1344729, 0.1577052
2: -0.1029525, 0.1605961, -0.0824195, 0.1223470, -0.2252996, 0.2430156
3: -0.0576805, 0.1386453, -0.0466005, 0.0980623, -0.1557429, 0.1852458
4: -0.1189080, 0.1891569, -0.0908781, 0.1427428, -0.2616507, 0.2800350

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
time: 0.31 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
time: 0.32 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0358978, 0.0521367, -0.0880345, 0.0880345
1: -0.0510239, 0.1157805, -0.0510239, 0.1157805, -0.1668044, 0.1668043
2: -0.1029525, 0.1605961, -0.1029525, 0.1605961, -0.2635486, 0.2635487
3: -0.0576805, 0.1386453, -0.0576805, 0.1386453, -0.1963258, 0.1963258
4: -0.1189080, 0.1891569, -0.1189080, 0.1891569, -0.3080648, 0.3080649

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317
time: 0.30 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.28 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.28
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0303923, 0.0316036, -0.0601504, 0.0592287
1: -0.0394259, 0.0744049, -0.0419247, 0.0834490, -0.1228749, 0.1163296
2: -0.0761525, 0.1130898, -0.0824195, 0.1223470, -0.1984996, 0.1955093
3: -0.0434286, 0.0854480, -0.0466005, 0.0980623, -0.1414909, 0.1320485
4: -0.0817791, 0.1311970, -0.0908781, 0.1427428, -0.2245218, 0.2220750

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0358978, 0.0521367, -0.0806835, 0.0647341
1: -0.0394259, 0.0744049, -0.0510239, 0.1157805, -0.1552063, 0.1254288
2: -0.0761525, 0.1130898, -0.1029525, 0.1605961, -0.2367487, 0.2160423
3: -0.0434286, 0.0854480, -0.0576805, 0.1386453, -0.1820739, 0.1431285
4: -0.0817791, 0.1311970, -0.1189080, 0.1891569, -0.2709360, 0.2501049

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0303923, 0.0316036, -0.0634249, 0.0704445
1: -0.0462006, 0.0969325, -0.0419247, 0.0834490, -0.1296496, 0.1388572
2: -0.0906157, 0.1402107, -0.0824195, 0.1223470, -0.2129627, 0.2226302
3: -0.0531606, 0.1144974, -0.0466005, 0.0980623, -0.1512229, 0.1610979
4: -0.1036385, 0.1661279, -0.0908781, 0.1427428, -0.2463813, 0.2570060

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0358978, 0.0521367, -0.0839581, 0.0759500
1: -0.0462006, 0.0969325, -0.0510239, 0.1157805, -0.1619810, 0.1479563
2: -0.0906157, 0.1402107, -0.1029525, 0.1605961, -0.2512118, 0.2431632
3: -0.0531606, 0.1144974, -0.0576805, 0.1386453, -0.1918058, 0.1721780
4: -0.1036385, 0.1661279, -0.1189080, 0.1891569, -0.2927953, 0.2850358

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.34 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
Binary search (step 11): status=Status.VERIFIED, low=0.1999512, high=0.2000000, mid=0.1999512, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 12) starts
Candidate diff: 0.1999756


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895600, upper bound: 0.0897790
time: 0.33 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781
time: 0.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.78 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -0.0895600, upper bound: 0.0897790
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0390887, 0.0634215, -0.0938137, 0.0706923
1: -0.0419247, 0.0834490, -0.0557757, 0.1414140, -0.1833387, 0.1392248
2: -0.0824195, 0.1223470, -0.1134923, 0.1820314, -0.2644509, 0.2358393
3: -0.0466005, 0.0980623, -0.0636387, 0.1747649, -0.2213654, 0.1617010
4: -0.0908781, 0.1427428, -0.1363116, 0.2151742, -0.3060522, 0.2790544

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
time: 0.32 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
time: 0.31 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0390887, 0.0634215, -0.0993192, 0.0912254
1: -0.0510239, 0.1157805, -0.0557757, 0.1414140, -0.1924379, 0.1715562
2: -0.1029525, 0.1605961, -0.1134923, 0.1820314, -0.2849840, 0.2740884
3: -0.0576805, 0.1386453, -0.0636387, 0.1747649, -0.2324454, 0.2022839
4: -0.1189080, 0.1891569, -0.1363116, 0.2151742, -0.3340822, 0.3254685

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
time: 0.30 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781
time: 0.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.48 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.48
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.48
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.48
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.48
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0303923, 0.0316036, -0.0619959, 0.0619959
1: -0.0419247, 0.0834490, -0.0419247, 0.0834490, -0.1253737, 0.1253737
2: -0.0824195, 0.1223470, -0.0824195, 0.1223470, -0.2047665, 0.2047665
3: -0.0466005, 0.0980623, -0.0466005, 0.0980623, -0.1446628, 0.1446628
4: -0.0908781, 0.1427428, -0.0908781, 0.1427428, -0.2336209, 0.2336209

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
time: 0.31 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0358978, 0.0521367, -0.0825290, 0.0675014
1: -0.0419247, 0.0834490, -0.0510239, 0.1157805, -0.1577052, 0.1344729
2: -0.0824195, 0.1223470, -0.1029525, 0.1605961, -0.2430156, 0.2252996
3: -0.0466005, 0.0980623, -0.0576805, 0.1386453, -0.1852457, 0.1557429
4: -0.0908781, 0.1427428, -0.1189080, 0.1891569, -0.2800349, 0.2616507

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
time: 0.31 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0303923, 0.0316036, -0.0675014, 0.0825290
1: -0.0510239, 0.1157805, -0.0419247, 0.0834490, -0.1344729, 0.1577052
2: -0.1029525, 0.1605961, -0.0824195, 0.1223470, -0.2252996, 0.2430156
3: -0.0576805, 0.1386453, -0.0466005, 0.0980623, -0.1557429, 0.1852458
4: -0.1189080, 0.1891569, -0.0908781, 0.1427428, -0.2616507, 0.2800350

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
time: 0.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0358978, 0.0521367, -0.0880345, 0.0880345
1: -0.0510239, 0.1157805, -0.0510239, 0.1157805, -0.1668044, 0.1668043
2: -0.1029525, 0.1605961, -0.1029525, 0.1605961, -0.2635486, 0.2635487
3: -0.0576805, 0.1386453, -0.0576805, 0.1386453, -0.1963258, 0.1963258
4: -0.1189080, 0.1891569, -0.1189080, 0.1891569, -0.3080648, 0.3080649

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317
time: 0.32 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.53 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.53
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0303923, 0.0316036, -0.0601504, 0.0592287
1: -0.0394259, 0.0744049, -0.0419247, 0.0834490, -0.1228749, 0.1163296
2: -0.0761525, 0.1130898, -0.0824195, 0.1223470, -0.1984996, 0.1955093
3: -0.0434286, 0.0854480, -0.0466005, 0.0980623, -0.1414909, 0.1320485
4: -0.0817791, 0.1311970, -0.0908781, 0.1427428, -0.2245218, 0.2220750

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0358978, 0.0521367, -0.0806835, 0.0647341
1: -0.0394259, 0.0744049, -0.0510239, 0.1157805, -0.1552063, 0.1254288
2: -0.0761525, 0.1130898, -0.1029525, 0.1605961, -0.2367487, 0.2160423
3: -0.0434286, 0.0854480, -0.0576805, 0.1386453, -0.1820739, 0.1431285
4: -0.0817791, 0.1311970, -0.1189080, 0.1891569, -0.2709360, 0.2501049

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0303923, 0.0316036, -0.0634249, 0.0704445
1: -0.0462006, 0.0969325, -0.0419247, 0.0834490, -0.1296496, 0.1388572
2: -0.0906157, 0.1402107, -0.0824195, 0.1223470, -0.2129627, 0.2226302
3: -0.0531606, 0.1144974, -0.0466005, 0.0980623, -0.1512229, 0.1610979
4: -0.1036385, 0.1661279, -0.0908781, 0.1427428, -0.2463813, 0.2570060

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0358978, 0.0521367, -0.0839581, 0.0759500
1: -0.0462006, 0.0969325, -0.0510239, 0.1157805, -0.1619810, 0.1479563
2: -0.0906157, 0.1402107, -0.1029525, 0.1605961, -0.2512118, 0.2431632
3: -0.0531606, 0.1144974, -0.0576805, 0.1386453, -0.1918058, 0.1721780
4: -0.1036385, 0.1661279, -0.1189080, 0.1891569, -0.2927953, 0.2850358

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.35 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.58 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.58
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
Binary search (step 12): status=Status.VERIFIED, low=0.1999756, high=0.2000000, mid=0.1999756, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 13) starts
Candidate diff: 0.1999878


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895600, upper bound: 0.0897790
time: 0.34 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781
time: 0.32 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.81 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.81
Output dim: 0, lower bound: -0.0895600, upper bound: 0.0897790
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.81
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0390887, 0.0634215, -0.0938137, 0.0706923
1: -0.0419247, 0.0834490, -0.0557757, 0.1414140, -0.1833387, 0.1392248
2: -0.0824195, 0.1223470, -0.1134923, 0.1820314, -0.2644509, 0.2358393
3: -0.0466005, 0.0980623, -0.0636387, 0.1747649, -0.2213654, 0.1617010
4: -0.0908781, 0.1427428, -0.1363116, 0.2151742, -0.3060522, 0.2790544

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
time: 0.30 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
time: 0.30 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0390887, 0.0634215, -0.0993192, 0.0912254
1: -0.0510239, 0.1157805, -0.0557757, 0.1414140, -0.1924379, 0.1715562
2: -0.1029525, 0.1605961, -0.1134923, 0.1820314, -0.2849840, 0.2740884
3: -0.0576805, 0.1386453, -0.0636387, 0.1747649, -0.2324454, 0.2022839
4: -0.1189080, 0.1891569, -0.1363116, 0.2151742, -0.3340822, 0.3254685

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
time: 0.29 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781
time: 0.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.23 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.23
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0303923, 0.0316036, -0.0619959, 0.0619959
1: -0.0419247, 0.0834490, -0.0419247, 0.0834490, -0.1253737, 0.1253737
2: -0.0824195, 0.1223470, -0.0824195, 0.1223470, -0.2047665, 0.2047665
3: -0.0466005, 0.0980623, -0.0466005, 0.0980623, -0.1446628, 0.1446628
4: -0.0908781, 0.1427428, -0.0908781, 0.1427428, -0.2336209, 0.2336209

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
time: 0.32 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
time: 0.30 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0358978, 0.0521367, -0.0825290, 0.0675014
1: -0.0419247, 0.0834490, -0.0510239, 0.1157805, -0.1577052, 0.1344729
2: -0.0824195, 0.1223470, -0.1029525, 0.1605961, -0.2430156, 0.2252996
3: -0.0466005, 0.0980623, -0.0576805, 0.1386453, -0.1852457, 0.1557429
4: -0.0908781, 0.1427428, -0.1189080, 0.1891569, -0.2800349, 0.2616507

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
time: 0.30 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
time: 0.30 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0303923, 0.0316036, -0.0675014, 0.0825290
1: -0.0510239, 0.1157805, -0.0419247, 0.0834490, -0.1344729, 0.1577052
2: -0.1029525, 0.1605961, -0.0824195, 0.1223470, -0.2252996, 0.2430156
3: -0.0576805, 0.1386453, -0.0466005, 0.0980623, -0.1557429, 0.1852458
4: -0.1189080, 0.1891569, -0.0908781, 0.1427428, -0.2616507, 0.2800350

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
time: 0.31 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
time: 0.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0358978, 0.0521367, -0.0880345, 0.0880345
1: -0.0510239, 0.1157805, -0.0510239, 0.1157805, -0.1668044, 0.1668043
2: -0.1029525, 0.1605961, -0.1029525, 0.1605961, -0.2635486, 0.2635487
3: -0.0576805, 0.1386453, -0.0576805, 0.1386453, -0.1963258, 0.1963258
4: -0.1189080, 0.1891569, -0.1189080, 0.1891569, -0.3080648, 0.3080649

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317
time: 0.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.34 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.34
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.34
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.34
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.34
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.34
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0303923, 0.0316036, -0.0601504, 0.0592287
1: -0.0394259, 0.0744049, -0.0419247, 0.0834490, -0.1228749, 0.1163296
2: -0.0761525, 0.1130898, -0.0824195, 0.1223470, -0.1984996, 0.1955093
3: -0.0434286, 0.0854480, -0.0466005, 0.0980623, -0.1414909, 0.1320485
4: -0.0817791, 0.1311970, -0.0908781, 0.1427428, -0.2245218, 0.2220750

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.32 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0358978, 0.0521367, -0.0806835, 0.0647341
1: -0.0394259, 0.0744049, -0.0510239, 0.1157805, -0.1552063, 0.1254288
2: -0.0761525, 0.1130898, -0.1029525, 0.1605961, -0.2367487, 0.2160423
3: -0.0434286, 0.0854480, -0.0576805, 0.1386453, -0.1820739, 0.1431285
4: -0.0817791, 0.1311970, -0.1189080, 0.1891569, -0.2709360, 0.2501049

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0303923, 0.0316036, -0.0634249, 0.0704445
1: -0.0462006, 0.0969325, -0.0419247, 0.0834490, -0.1296496, 0.1388572
2: -0.0906157, 0.1402107, -0.0824195, 0.1223470, -0.2129627, 0.2226302
3: -0.0531606, 0.1144974, -0.0466005, 0.0980623, -0.1512229, 0.1610979
4: -0.1036385, 0.1661279, -0.0908781, 0.1427428, -0.2463813, 0.2570060

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.31 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0358978, 0.0521367, -0.0839581, 0.0759500
1: -0.0462006, 0.0969325, -0.0510239, 0.1157805, -0.1619810, 0.1479563
2: -0.0906157, 0.1402107, -0.1029525, 0.1605961, -0.2512118, 0.2431632
3: -0.0531606, 0.1144974, -0.0576805, 0.1386453, -0.1918058, 0.1721780
4: -0.1036385, 0.1661279, -0.1189080, 0.1891569, -0.2927953, 0.2850358

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.34 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
Binary search (step 13): status=Status.VERIFIED, low=0.1999878, high=0.2000000, mid=0.1999878, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 14) starts
Candidate diff: 0.1999939


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895600, upper bound: 0.0897790
time: 0.33 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781
time: 0.31 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.78 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -0.0895600, upper bound: 0.0897790
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.78
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0390887, 0.0634215, -0.0938137, 0.0706923
1: -0.0419247, 0.0834490, -0.0557757, 0.1414140, -0.1833387, 0.1392248
2: -0.0824195, 0.1223470, -0.1134923, 0.1820314, -0.2644509, 0.2358393
3: -0.0466005, 0.0980623, -0.0636387, 0.1747649, -0.2213654, 0.1617010
4: -0.0908781, 0.1427428, -0.1363116, 0.2151742, -0.3060522, 0.2790544

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
time: 0.30 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
time: 0.30 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0390887, 0.0634215, -0.0993192, 0.0912254
1: -0.0510239, 0.1157805, -0.0557757, 0.1414140, -0.1924379, 0.1715562
2: -0.1029525, 0.1605961, -0.1134923, 0.1820314, -0.2849840, 0.2740884
3: -0.0576805, 0.1386453, -0.0636387, 0.1747649, -0.2324454, 0.2022839
4: -0.1189080, 0.1891569, -0.1363116, 0.2151742, -0.3340822, 0.3254685

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
time: 0.29 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781
time: 0.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.19 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.19
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0303923, 0.0316036, -0.0619959, 0.0619959
1: -0.0419247, 0.0834490, -0.0419247, 0.0834490, -0.1253737, 0.1253737
2: -0.0824195, 0.1223470, -0.0824195, 0.1223470, -0.2047665, 0.2047665
3: -0.0466005, 0.0980623, -0.0466005, 0.0980623, -0.1446628, 0.1446628
4: -0.0908781, 0.1427428, -0.0908781, 0.1427428, -0.2336209, 0.2336209

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
time: 0.32 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0358978, 0.0521367, -0.0825290, 0.0675014
1: -0.0419247, 0.0834490, -0.0510239, 0.1157805, -0.1577052, 0.1344729
2: -0.0824195, 0.1223470, -0.1029525, 0.1605961, -0.2430156, 0.2252996
3: -0.0466005, 0.0980623, -0.0576805, 0.1386453, -0.1852457, 0.1557429
4: -0.0908781, 0.1427428, -0.1189080, 0.1891569, -0.2800349, 0.2616507

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
time: 0.33 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0303923, 0.0316036, -0.0675014, 0.0825290
1: -0.0510239, 0.1157805, -0.0419247, 0.0834490, -0.1344729, 0.1577052
2: -0.1029525, 0.1605961, -0.0824195, 0.1223470, -0.2252996, 0.2430156
3: -0.0576805, 0.1386453, -0.0466005, 0.0980623, -0.1557429, 0.1852458
4: -0.1189080, 0.1891569, -0.0908781, 0.1427428, -0.2616507, 0.2800350

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
time: 0.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0358978, 0.0521367, -0.0880345, 0.0880345
1: -0.0510239, 0.1157805, -0.0510239, 0.1157805, -0.1668044, 0.1668043
2: -0.1029525, 0.1605961, -0.1029525, 0.1605961, -0.2635486, 0.2635487
3: -0.0576805, 0.1386453, -0.0576805, 0.1386453, -0.1963258, 0.1963258
4: -0.1189080, 0.1891569, -0.1189080, 0.1891569, -0.3080648, 0.3080649

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317
time: 0.33 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.52 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.52
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0303923, 0.0316036, -0.0601504, 0.0592287
1: -0.0394259, 0.0744049, -0.0419247, 0.0834490, -0.1228749, 0.1163296
2: -0.0761525, 0.1130898, -0.0824195, 0.1223470, -0.1984996, 0.1955093
3: -0.0434286, 0.0854480, -0.0466005, 0.0980623, -0.1414909, 0.1320485
4: -0.0817791, 0.1311970, -0.0908781, 0.1427428, -0.2245218, 0.2220750

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.31 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0358978, 0.0521367, -0.0806835, 0.0647341
1: -0.0394259, 0.0744049, -0.0510239, 0.1157805, -0.1552063, 0.1254288
2: -0.0761525, 0.1130898, -0.1029525, 0.1605961, -0.2367487, 0.2160423
3: -0.0434286, 0.0854480, -0.0576805, 0.1386453, -0.1820739, 0.1431285
4: -0.0817791, 0.1311970, -0.1189080, 0.1891569, -0.2709360, 0.2501049

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0303923, 0.0316036, -0.0634249, 0.0704445
1: -0.0462006, 0.0969325, -0.0419247, 0.0834490, -0.1296496, 0.1388572
2: -0.0906157, 0.1402107, -0.0824195, 0.1223470, -0.2129627, 0.2226302
3: -0.0531606, 0.1144974, -0.0466005, 0.0980623, -0.1512229, 0.1610979
4: -0.1036385, 0.1661279, -0.0908781, 0.1427428, -0.2463813, 0.2570060

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0358978, 0.0521367, -0.0839581, 0.0759500
1: -0.0462006, 0.0969325, -0.0510239, 0.1157805, -0.1619810, 0.1479563
2: -0.0906157, 0.1402107, -0.1029525, 0.1605961, -0.2512118, 0.2431632
3: -0.0531606, 0.1144974, -0.0576805, 0.1386453, -0.1918058, 0.1721780
4: -0.1036385, 0.1661279, -0.1189080, 0.1891569, -0.2927953, 0.2850358

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.35 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.59 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.59
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
Binary search (step 14): status=Status.VERIFIED, low=0.1999939, high=0.2000000, mid=0.1999939, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 15) starts
Candidate diff: 0.1999969


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895600, upper bound: 0.0897790
time: 0.35 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781
time: 0.33 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.84 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.84
Output dim: 0, lower bound: -0.0895600, upper bound: 0.0897790
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.84
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0390887, 0.0634215, -0.0938137, 0.0706923
1: -0.0419247, 0.0834490, -0.0557757, 0.1414140, -0.1833387, 0.1392248
2: -0.0824195, 0.1223470, -0.1134923, 0.1820314, -0.2644509, 0.2358393
3: -0.0466005, 0.0980623, -0.0636387, 0.1747649, -0.2213654, 0.1617010
4: -0.0908781, 0.1427428, -0.1363116, 0.2151742, -0.3060522, 0.2790544

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
time: 0.31 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
time: 0.31 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0390887, 0.0634215, -0.0993192, 0.0912254
1: -0.0510239, 0.1157805, -0.0557757, 0.1414140, -0.1924379, 0.1715562
2: -0.1029525, 0.1605961, -0.1134923, 0.1820314, -0.2849840, 0.2740884
3: -0.0576805, 0.1386453, -0.0636387, 0.1747649, -0.2324454, 0.2022839
4: -0.1189080, 0.1891569, -0.1363116, 0.2151742, -0.3340822, 0.3254685

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
time: 0.31 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781
time: 0.31 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.54 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.54
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0303923, 0.0316036, -0.0619959, 0.0619959
1: -0.0419247, 0.0834490, -0.0419247, 0.0834490, -0.1253737, 0.1253737
2: -0.0824195, 0.1223470, -0.0824195, 0.1223470, -0.2047665, 0.2047665
3: -0.0466005, 0.0980623, -0.0466005, 0.0980623, -0.1446628, 0.1446628
4: -0.0908781, 0.1427428, -0.0908781, 0.1427428, -0.2336209, 0.2336209

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
time: 0.34 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
time: 0.31 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0358978, 0.0521367, -0.0825290, 0.0675014
1: -0.0419247, 0.0834490, -0.0510239, 0.1157805, -0.1577052, 0.1344729
2: -0.0824195, 0.1223470, -0.1029525, 0.1605961, -0.2430156, 0.2252996
3: -0.0466005, 0.0980623, -0.0576805, 0.1386453, -0.1852457, 0.1557429
4: -0.0908781, 0.1427428, -0.1189080, 0.1891569, -0.2800349, 0.2616507

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
time: 0.30 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0303923, 0.0316036, -0.0675014, 0.0825290
1: -0.0510239, 0.1157805, -0.0419247, 0.0834490, -0.1344729, 0.1577052
2: -0.1029525, 0.1605961, -0.0824195, 0.1223470, -0.2252996, 0.2430156
3: -0.0576805, 0.1386453, -0.0466005, 0.0980623, -0.1557429, 0.1852458
4: -0.1189080, 0.1891569, -0.0908781, 0.1427428, -0.2616507, 0.2800350

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
time: 0.31 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
time: 0.33 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0358978, 0.0521367, -0.0880345, 0.0880345
1: -0.0510239, 0.1157805, -0.0510239, 0.1157805, -0.1668044, 0.1668043
2: -0.1029525, 0.1605961, -0.1029525, 0.1605961, -0.2635486, 0.2635487
3: -0.0576805, 0.1386453, -0.0576805, 0.1386453, -0.1963258, 0.1963258
4: -0.1189080, 0.1891569, -0.1189080, 0.1891569, -0.3080648, 0.3080649

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317
time: 0.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.43 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.43
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0303923, 0.0316036, -0.0601504, 0.0592287
1: -0.0394259, 0.0744049, -0.0419247, 0.0834490, -0.1228749, 0.1163296
2: -0.0761525, 0.1130898, -0.0824195, 0.1223470, -0.1984996, 0.1955093
3: -0.0434286, 0.0854480, -0.0466005, 0.0980623, -0.1414909, 0.1320485
4: -0.0817791, 0.1311970, -0.0908781, 0.1427428, -0.2245218, 0.2220750

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.32 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0358978, 0.0521367, -0.0806835, 0.0647341
1: -0.0394259, 0.0744049, -0.0510239, 0.1157805, -0.1552063, 0.1254288
2: -0.0761525, 0.1130898, -0.1029525, 0.1605961, -0.2367487, 0.2160423
3: -0.0434286, 0.0854480, -0.0576805, 0.1386453, -0.1820739, 0.1431285
4: -0.0817791, 0.1311970, -0.1189080, 0.1891569, -0.2709360, 0.2501049

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0303923, 0.0316036, -0.0634249, 0.0704445
1: -0.0462006, 0.0969325, -0.0419247, 0.0834490, -0.1296496, 0.1388572
2: -0.0906157, 0.1402107, -0.0824195, 0.1223470, -0.2129627, 0.2226302
3: -0.0531606, 0.1144974, -0.0466005, 0.0980623, -0.1512229, 0.1610979
4: -0.1036385, 0.1661279, -0.0908781, 0.1427428, -0.2463813, 0.2570060

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.31 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0358978, 0.0521367, -0.0839581, 0.0759500
1: -0.0462006, 0.0969325, -0.0510239, 0.1157805, -0.1619810, 0.1479563
2: -0.0906157, 0.1402107, -0.1029525, 0.1605961, -0.2512118, 0.2431632
3: -0.0531606, 0.1144974, -0.0576805, 0.1386453, -0.1918058, 0.1721780
4: -0.1036385, 0.1661279, -0.1189080, 0.1891569, -0.2927953, 0.2850358

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.33 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.33
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
Binary search (step 15): status=Status.VERIFIED, low=0.1999969, high=0.2000000, mid=0.1999969, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 16) starts
Candidate diff: 0.1999985


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895600, upper bound: 0.0897790
time: 0.33 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781
time: 0.31 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.79 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -0.0895600, upper bound: 0.0897790
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0390887, 0.0634215, -0.0938137, 0.0706923
1: -0.0419247, 0.0834490, -0.0557757, 0.1414140, -0.1833387, 0.1392248
2: -0.0824195, 0.1223470, -0.1134923, 0.1820314, -0.2644509, 0.2358393
3: -0.0466005, 0.0980623, -0.0636387, 0.1747649, -0.2213654, 0.1617010
4: -0.0908781, 0.1427428, -0.1363116, 0.2151742, -0.3060522, 0.2790544

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
time: 0.30 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
time: 0.30 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0390887, 0.0634215, -0.0993192, 0.0912254
1: -0.0510239, 0.1157805, -0.0557757, 0.1414140, -0.1924379, 0.1715562
2: -0.1029525, 0.1605961, -0.1134923, 0.1820314, -0.2849840, 0.2740884
3: -0.0576805, 0.1386453, -0.0636387, 0.1747649, -0.2324454, 0.2022839
4: -0.1189080, 0.1891569, -0.1363116, 0.2151742, -0.3340822, 0.3254685

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
time: 0.29 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781
time: 0.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.27 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.27
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0303923, 0.0316036, -0.0619959, 0.0619959
1: -0.0419247, 0.0834490, -0.0419247, 0.0834490, -0.1253737, 0.1253737
2: -0.0824195, 0.1223470, -0.0824195, 0.1223470, -0.2047665, 0.2047665
3: -0.0466005, 0.0980623, -0.0466005, 0.0980623, -0.1446628, 0.1446628
4: -0.0908781, 0.1427428, -0.0908781, 0.1427428, -0.2336209, 0.2336209

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
time: 0.31 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0358978, 0.0521367, -0.0825290, 0.0675014
1: -0.0419247, 0.0834490, -0.0510239, 0.1157805, -0.1577052, 0.1344729
2: -0.0824195, 0.1223470, -0.1029525, 0.1605961, -0.2430156, 0.2252996
3: -0.0466005, 0.0980623, -0.0576805, 0.1386453, -0.1852457, 0.1557429
4: -0.0908781, 0.1427428, -0.1189080, 0.1891569, -0.2800349, 0.2616507

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
time: 0.30 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
time: 0.32 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0303923, 0.0316036, -0.0675014, 0.0825290
1: -0.0510239, 0.1157805, -0.0419247, 0.0834490, -0.1344729, 0.1577052
2: -0.1029525, 0.1605961, -0.0824195, 0.1223470, -0.2252996, 0.2430156
3: -0.0576805, 0.1386453, -0.0466005, 0.0980623, -0.1557429, 0.1852458
4: -0.1189080, 0.1891569, -0.0908781, 0.1427428, -0.2616507, 0.2800350

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
time: 0.34 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
time: 0.34 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0358978, 0.0521367, -0.0880345, 0.0880345
1: -0.0510239, 0.1157805, -0.0510239, 0.1157805, -0.1668044, 0.1668043
2: -0.1029525, 0.1605961, -0.1029525, 0.1605961, -0.2635486, 0.2635487
3: -0.0576805, 0.1386453, -0.0576805, 0.1386453, -0.1963258, 0.1963258
4: -0.1189080, 0.1891569, -0.1189080, 0.1891569, -0.3080648, 0.3080649

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
time: 0.33 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317
time: 0.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.57 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.57
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.57
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.57
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.57
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.57
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0303923, 0.0316036, -0.0601504, 0.0592287
1: -0.0394259, 0.0744049, -0.0419247, 0.0834490, -0.1228749, 0.1163296
2: -0.0761525, 0.1130898, -0.0824195, 0.1223470, -0.1984996, 0.1955093
3: -0.0434286, 0.0854480, -0.0466005, 0.0980623, -0.1414909, 0.1320485
4: -0.0817791, 0.1311970, -0.0908781, 0.1427428, -0.2245218, 0.2220750

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.34 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0358978, 0.0521367, -0.0806835, 0.0647341
1: -0.0394259, 0.0744049, -0.0510239, 0.1157805, -0.1552063, 0.1254288
2: -0.0761525, 0.1130898, -0.1029525, 0.1605961, -0.2367487, 0.2160423
3: -0.0434286, 0.0854480, -0.0576805, 0.1386453, -0.1820739, 0.1431285
4: -0.0817791, 0.1311970, -0.1189080, 0.1891569, -0.2709360, 0.2501049

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0303923, 0.0316036, -0.0634249, 0.0704445
1: -0.0462006, 0.0969325, -0.0419247, 0.0834490, -0.1296496, 0.1388572
2: -0.0906157, 0.1402107, -0.0824195, 0.1223470, -0.2129627, 0.2226302
3: -0.0531606, 0.1144974, -0.0466005, 0.0980623, -0.1512229, 0.1610979
4: -0.1036385, 0.1661279, -0.0908781, 0.1427428, -0.2463813, 0.2570060

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0358978, 0.0521367, -0.0839581, 0.0759500
1: -0.0462006, 0.0969325, -0.0510239, 0.1157805, -0.1619810, 0.1479563
2: -0.0906157, 0.1402107, -0.1029525, 0.1605961, -0.2512118, 0.2431632
3: -0.0531606, 0.1144974, -0.0576805, 0.1386453, -0.1918058, 0.1721780
4: -0.1036385, 0.1661279, -0.1189080, 0.1891569, -0.2927953, 0.2850358

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.36 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.68 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
Binary search (step 16): status=Status.VERIFIED, low=0.1999985, high=0.2000000, mid=0.1999985, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary search (step 17) starts
Candidate diff: 0.1999992


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895600, upper bound: 0.0897790
time: 0.30 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781
time: 0.34 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.80 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -0.0895600, upper bound: 0.0897790
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.80
Output dim: 0, lower bound: -0.0895781, upper bound: 0.0895781

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0390887, 0.0634215, -0.0938137, 0.0706923
1: -0.0419247, 0.0834490, -0.0557757, 0.1414140, -0.1833387, 0.1392248
2: -0.0824195, 0.1223470, -0.1134923, 0.1820314, -0.2644509, 0.2358393
3: -0.0466005, 0.0980623, -0.0636387, 0.1747649, -0.2213654, 0.1617010
4: -0.0908781, 0.1427428, -0.1363116, 0.2151742, -0.3060522, 0.2790544

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
time: 0.31 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
time: 0.33 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0390887, 0.0634215, -0.0993192, 0.0912254
1: -0.0510239, 0.1157805, -0.0557757, 0.1414140, -0.1924379, 0.1715562
2: -0.1029525, 0.1605961, -0.1134923, 0.1820314, -0.2849840, 0.2740884
3: -0.0576805, 0.1386453, -0.0636387, 0.1747649, -0.2324454, 0.2022839
4: -0.1189080, 0.1891569, -0.1363116, 0.2151742, -0.3340822, 0.3254685

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
time: 0.31 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781
time: 0.31 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.53 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0890596
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 0, lower bound: -0.0890596, upper bound: 0.0891106
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895271
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.53
Output dim: 0, lower bound: -0.0891106, upper bound: 0.0895781

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0303923, 0.0316036, -0.0619959, 0.0619959
1: -0.0419247, 0.0834490, -0.0419247, 0.0834490, -0.1253737, 0.1253737
2: -0.0824195, 0.1223470, -0.0824195, 0.1223470, -0.2047665, 0.2047665
3: -0.0466005, 0.0980623, -0.0466005, 0.0980623, -0.1446628, 0.1446628
4: -0.0908781, 0.1427428, -0.0908781, 0.1427428, -0.2336209, 0.2336209

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
time: 0.33 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
time: 0.31 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0303923, 0.0316036, -0.0358978, 0.0521367, -0.0825290, 0.0675014
1: -0.0419247, 0.0834490, -0.0510239, 0.1157805, -0.1577052, 0.1344729
2: -0.0824195, 0.1223470, -0.1029525, 0.1605961, -0.2430156, 0.2252996
3: -0.0466005, 0.0980623, -0.0576805, 0.1386453, -0.1852457, 0.1557429
4: -0.0908781, 0.1427428, -0.1189080, 0.1891569, -0.2800349, 0.2616507

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
time: 0.33 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0303923, 0.0316036, -0.0675014, 0.0825290
1: -0.0510239, 0.1157805, -0.0419247, 0.0834490, -0.1344729, 0.1577052
2: -0.1029525, 0.1605961, -0.0824195, 0.1223470, -0.2252996, 0.2430156
3: -0.0576805, 0.1386453, -0.0466005, 0.0980623, -0.1557429, 0.1852458
4: -0.1189080, 0.1891569, -0.0908781, 0.1427428, -0.2616507, 0.2800350

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
time: 0.35 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
time: 0.32 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0358978, 0.0521367, -0.0358978, 0.0521367, -0.0880345, 0.0880345
1: -0.0510239, 0.1157805, -0.0510239, 0.1157805, -0.1668044, 0.1668043
2: -0.1029525, 0.1605961, -0.1029525, 0.1605961, -0.2635486, 0.2635487
3: -0.0576805, 0.1386453, -0.0576805, 0.1386453, -0.1963258, 0.1963258
4: -0.1189080, 0.1891569, -0.1189080, 0.1891569, -0.3080648, 0.3080649

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
time: 0.32 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317
time: 0.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.33 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.33
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879144
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -0.0880529, upper bound: 0.0894954
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.33
Output dim: 0, lower bound: -0.0878510, upper bound: 0.0879593
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893028
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.33
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0879872
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.33
Output dim: 0, lower bound: -0.0880675, upper bound: 0.0893476
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.33
Output dim: 0, lower bound: -0.0878958, upper bound: 0.0880317

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0303923, 0.0316036, -0.0601504, 0.0592287
1: -0.0394259, 0.0744049, -0.0419247, 0.0834490, -0.1228749, 0.1163296
2: -0.0761525, 0.1130898, -0.0824195, 0.1223470, -0.1984996, 0.1955093
3: -0.0434286, 0.0854480, -0.0466005, 0.0980623, -0.1414909, 0.1320485
4: -0.0817791, 0.1311970, -0.0908781, 0.1427428, -0.2245218, 0.2220750

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
time: 0.32 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0285468, 0.0288364, -0.0358978, 0.0521367, -0.0806835, 0.0647341
1: -0.0394259, 0.0744049, -0.0510239, 0.1157805, -0.1552063, 0.1254288
2: -0.0761525, 0.1130898, -0.1029525, 0.1605961, -0.2367487, 0.2160423
3: -0.0434286, 0.0854480, -0.0576805, 0.1386453, -0.1820739, 0.1431285
4: -0.0817791, 0.1311970, -0.1189080, 0.1891569, -0.2709360, 0.2501049

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
time: 0.33 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0303923, 0.0316036, -0.0634249, 0.0704445
1: -0.0462006, 0.0969325, -0.0419247, 0.0834490, -0.1296496, 0.1388572
2: -0.0906157, 0.1402107, -0.0824195, 0.1223470, -0.2129627, 0.2226302
3: -0.0531606, 0.1144974, -0.0466005, 0.0980623, -0.1512229, 0.1610979
4: -0.1036385, 0.1661279, -0.0908781, 0.1427428, -0.2463813, 0.2570060

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.40 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0318214, 0.0400522, -0.0358978, 0.0521367, -0.0839581, 0.0759500
1: -0.0462006, 0.0969325, -0.0510239, 0.1157805, -0.1619810, 0.1479563
2: -0.0906157, 0.1402107, -0.1029525, 0.1605961, -0.2512118, 0.2431632
3: -0.0531606, 0.1144974, -0.0576805, 0.1386453, -0.1918058, 0.1721780
4: -0.1036385, 0.1661279, -0.1189080, 0.1891569, -0.2927953, 0.2850358

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
time: 0.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.34 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0879144, upper bound: 0.0879144
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0879872, upper bound: 0.0879592
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0879592, upper bound: 0.0879872
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.34
Output dim: 0, lower bound: -0.0880320, upper bound: 0.0880317
Binary search (step 17): status=Status.VERIFIED, low=0.1999992, high=0.2000000, mid=0.1999992, abs_max=0.10251016169786453
rel_dist={0: [-0.08998774102526627, 0.08998774102526627]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.1999992251396634
execution time: 493.11 seconds
