## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 0)
Time budget: 420 seconds
Split limit: 100
Threshold: 3656.1557913764673


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1750.0124512, 3321.1867676, -1750.0124512, 3321.1867676, -5071.1992188, 5071.1992188)
1: (-591.4364624, 1249.1158447, -591.4364624, 1249.1158447, -1840.5522461, 1840.5522461)
2: (-302.9573364, 1257.5653076, -302.9573364, 1257.5653076, -1560.5225830, 1560.5225830)
3: (-690.8545532, 1531.0914307, -690.8545532, 1531.0914307, -2221.9460449, 2221.9460449)
4: (-393.2388916, 1293.9045410, -393.2388916, 1293.9045410, -1687.1434326, 1687.1434326)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.06 + 2.02 = 4.09 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -3656.1923533, upper bound: 3656.1923533

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1641240, upper bound: 3656.1540650
time: 0.73 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1923533, upper bound: 3656.1923533
time: 0.82 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.73 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.73
Output dim: 0, lower bound: -3656.1641240, upper bound: 3656.1540650
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.73
Output dim: 0, lower bound: -3656.1923533, upper bound: 3656.1923533

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -1749.5305176, 3319.4211426, -1749.7252197, 3320.6281738, -5070.1586914, 5069.1464844
1: -591.2507324, 1248.8815918, -591.3413696, 1248.9106445, -1840.1613770, 1840.2229004
2: -302.7809448, 1257.0837402, -302.9073792, 1257.3552246, -1560.1362305, 1559.9909668
3: -690.3079224, 1530.6922607, -690.7354736, 1530.8371582, -2221.1450195, 2221.4274902
4: -393.0486755, 1292.8083496, -393.1747742, 1293.6766357, -1686.7252197, 1685.9831543

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3656.0995421, upper bound: 3656.1235564
time: 0.73 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3656.0974993, upper bound: 3656.1127495
time: 0.78 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -1748.7803955, 3318.8728027, -1750.0124512, 3321.1867676, -5069.9672852, 5068.8852539
1: -591.0313721, 1248.2449951, -591.4364624, 1249.1158447, -1840.1472168, 1839.6813965
2: -302.7409058, 1256.6921387, -302.9573364, 1257.5653076, -1560.3059082, 1559.6494141
3: -690.3617554, 1530.0263672, -690.8545532, 1531.0914307, -2221.4531250, 2220.8808594
4: -392.9594727, 1292.9818115, -393.2388916, 1293.9045410, -1686.8640137, 1686.2207031

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1735215, upper bound: 3656.1676616
time: 0.76 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1923451, upper bound: 3656.1923451
time: 0.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.60 seconds
NS_A1_A1, status: Status.VERIFIED, split count: 2, time: 3.60
Output dim: 0, lower bound: -3656.0995421, upper bound: 3656.1235564
NS_A1_A2, status: Status.VERIFIED, split count: 2, time: 3.60
Output dim: 0, lower bound: -3656.0974993, upper bound: 3656.1127495
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 3.60
Output dim: 0, lower bound: -3656.1735215, upper bound: 3656.1676616
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 3.60
Output dim: 0, lower bound: -3656.1923451, upper bound: 3656.1923451

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -1682.4653320, 3206.7590332, -1734.9082031, 3294.0971680, -4976.5625000, 4941.6665039
1: -570.0602417, 1204.9034424, -586.5425415, 1238.9145508, -1808.9747314, 1791.4459229
2: -291.8394165, 1213.3343506, -300.4112854, 1247.2947998, -1539.1342773, 1513.7456055
3: -666.5479126, 1477.3549805, -685.1442871, 1518.6076660, -2185.1552734, 2162.4992676
4: -378.9655762, 1248.8038330, -389.9661865, 1283.2738037, -1662.2392578, 1638.7700195

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1625054, upper bound: 3656.1622517
time: 0.75 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1730075, upper bound: 3656.1666353
time: 0.77 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -1745.3035889, 3312.5268555, -1750.0124512, 3321.1867676, -5066.4897461, 5062.5390625
1: -589.8759766, 1245.7950439, -591.4364624, 1249.1158447, -1838.9918213, 1837.2314453
2: -302.1513977, 1254.2524414, -302.9573364, 1257.5653076, -1559.7166748, 1557.2097168
3: -689.0215454, 1527.0477295, -690.8545532, 1531.0914307, -2220.1127930, 2217.9023438
4: -392.1912842, 1290.4990234, -393.2388916, 1293.9045410, -1686.0958252, 1683.7379150

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_A2_A1

### Relational analysis result of NS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1902150, upper bound: 3656.1901771
time: 0.72 seconds

## Relational analysis of NS_A2_A2_A2

### Relational analysis result of NS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1896354, upper bound: 3656.1896354
time: 0.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.53 seconds
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 0, lower bound: -3656.1625054, upper bound: 3656.1622517
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 0, lower bound: -3656.1730075, upper bound: 3656.1666353
NS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 0, lower bound: -3656.1902150, upper bound: 3656.1901771
NS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.53
Output dim: 0, lower bound: -3656.1896354, upper bound: 3656.1896354

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -1641.4968262, 3133.1918945, -1581.0743408, 3021.0222168, -4662.5190430, 4714.2661133
1: -556.6383057, 1176.3253174, -536.1956177, 1131.9337158, -1688.5720215, 1712.5209961
2: -284.8349915, 1184.7612305, -274.0758972, 1140.7347412, -1425.5694580, 1458.8370361
3: -650.5821533, 1442.2714844, -625.6157837, 1387.8233643, -2038.4055176, 2067.8869629
4: -369.7657166, 1219.3765869, -355.4192200, 1174.3052979, -1544.0710449, 1574.7957764

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_A1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1625054, upper bound: 3656.1622517
time: 0.69 seconds

## Relational analysis of NS_A2_A1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1625054, upper bound: 3656.1622517
time: 0.73 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -1682.4653320, 3206.7590332, -1728.8958740, 3283.0593262, -4965.5244141, 4935.6542969
1: -570.0602417, 1204.9034424, -584.5508423, 1234.6733398, -1804.7335205, 1789.4543457
2: -291.8394165, 1213.3343506, -299.4026794, 1243.0773926, -1534.9167480, 1512.7370605
3: -666.5479126, 1477.3549805, -682.8367310, 1513.4703369, -2180.0183105, 2160.1916504
4: -378.9655762, 1248.8038330, -388.6503296, 1278.9775391, -1657.9429932, 1637.4541016

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1612432, upper bound: 3656.1612432
time: 0.68 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1612432, upper bound: 3656.1666353
time: 0.71 seconds

## BFS NS instance: NS_A2_A2_A1

### Backsubstitution after applying NS history:
0: -1723.9946289, 3273.3566895, -1744.3919678, 3310.8337402, -5034.8281250, 5017.7485352
1: -582.9691162, 1230.9879150, -589.6126099, 1245.2062988, -1828.1754150, 1820.6005859
2: -298.5708618, 1239.3637695, -302.0122375, 1253.6330566, -1552.2038574, 1541.3759766
3: -680.9635010, 1508.8701172, -688.7265015, 1526.2907715, -2207.2543945, 2197.5961914
4: -387.5766907, 1275.2492676, -392.0205688, 1289.8767090, -1677.4533691, 1667.2697754

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_A2_A1_B1

### Relational analysis result of NS_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1896354, upper bound: 3656.1896354
time: 0.71 seconds

## Relational analysis of NS_A2_A2_A1_B2

### Relational analysis result of NS_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1896354, upper bound: 3656.1896354
time: 0.82 seconds

## BFS NS instance: NS_A2_A2_A2

### Backsubstitution after applying NS history:
0: -1681.5596924, 3190.1123047, -1721.8073730, 3268.3962402, -4949.9550781, 4911.9199219
1: -567.9144287, 1198.9532471, -581.9077148, 1228.7713623, -1796.6857910, 1780.8609619
2: -290.9331665, 1207.3736572, -298.0340271, 1237.3323975, -1528.2656250, 1505.4077148
3: -663.3439941, 1469.7086182, -679.6489868, 1506.2297363, -2169.5737305, 2149.3574219
4: -377.6191406, 1242.4111328, -386.8245544, 1273.1224365, -1650.7414551, 1629.2354736

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_A2_A2_B1

### Relational analysis result of NS_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1896354, upper bound: 3656.1896354
time: 0.83 seconds

## Relational analysis of NS_A2_A2_A2_B2

### Relational analysis result of NS_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1896354, upper bound: 3656.1896354
time: 0.74 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.67 seconds
NS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -3656.1625054, upper bound: 3656.1622517
NS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -3656.1625054, upper bound: 3656.1622517
NS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -3656.1612432, upper bound: 3656.1612432
NS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -3656.1612432, upper bound: 3656.1666353
NS_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -3656.1896354, upper bound: 3656.1896354
NS_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -3656.1896354, upper bound: 3656.1896354
NS_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -3656.1896354, upper bound: 3656.1896354
NS_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.67
Output dim: 0, lower bound: -3656.1896354, upper bound: 3656.1896354

## BFS NS instance: NS_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1528.8487549, 2936.8110352, -1581.0743408, 3021.0222168, -4549.8696289, 4517.8852539
1: -520.2947388, 1098.7904053, -536.1956177, 1131.9337158, -1652.2285156, 1634.9860840
2: -265.7568970, 1107.6166992, -274.0758972, 1140.7347412, -1406.4914551, 1381.6925049
3: -607.3670044, 1347.3883057, -625.6157837, 1387.8233643, -1995.1901855, 1973.0041504
4: -344.7153320, 1141.4797363, -355.4192200, 1174.3052979, -1519.0205078, 1496.8988037

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_A1_B1_A1_B1

### Relational analysis result of NS_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1612855, upper bound: 3656.1615208
time: 0.77 seconds

## Relational analysis of NS_A2_A1_B1_A1_B2

### Relational analysis result of NS_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1612855, upper bound: 3656.1622517
time: 0.72 seconds

## BFS NS instance: NS_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1676.4122314, 3195.7221680, -1581.0743408, 3021.0222168, -4697.4340820, 4776.7963867
1: -568.0551147, 1200.6782227, -536.1956177, 1131.9337158, -1699.9887695, 1736.8737793
2: -290.8211365, 1209.1031494, -274.0758972, 1140.7347412, -1431.5556641, 1483.1790771
3: -664.2241821, 1472.2062988, -625.6157837, 1387.8233643, -2052.0471191, 2097.8217773
4: -377.6401672, 1244.4934082, -355.4192200, 1174.3052979, -1551.9454346, 1599.9125977

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_A1_B1_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1612855, upper bound: 3656.1615208
time: 0.78 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1612855, upper bound: 3656.1622517
time: 0.80 seconds

## BFS NS instance: NS_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -1682.4653320, 3206.7590332, -1677.6610107, 3198.0446777, -4880.5092773, 4884.4199219
1: -570.0602417, 1204.9034424, -568.4644165, 1201.5491943, -1771.6093750, 1773.3679199
2: -291.8394165, 1213.3343506, -291.0398560, 1209.9788818, -1501.8183594, 1504.3742676
3: -666.5479126, 1477.3549805, -664.7223511, 1473.2760010, -2139.8239746, 2142.0773926
4: -378.9655762, 1248.8038330, -377.9221802, 1245.4234619, -1624.3889160, 1626.7260742

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_A1_B2_B1_A1

### Relational analysis result of NS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1612432, upper bound: 3656.1612432
time: 0.78 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2

### Relational analysis result of NS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1612432, upper bound: 3656.1612432
time: 0.69 seconds

## BFS NS instance: NS_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -1682.4653320, 3206.7590332, -1740.6629639, 3304.0895996, -4986.5546875, 4947.4218750
1: -570.0602417, 1204.9034424, -588.3560791, 1242.5281982, -1812.5883789, 1793.2595215
2: -291.8394165, 1213.3343506, -301.3850403, 1251.0102539, -1542.8496094, 1514.7192383
3: -666.5479126, 1477.3549805, -687.2623901, 1523.1033936, -2189.6511230, 2164.6174316
4: -378.9655762, 1248.8038330, -391.1883850, 1287.2303467, -1666.1958008, 1639.9921875

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_A1_B2_B2_A1

### Relational analysis result of NS_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1612432, upper bound: 3656.1666350
time: 0.95 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2

### Relational analysis result of NS_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1612432, upper bound: 3656.1622591
time: 0.76 seconds

## BFS NS instance: NS_A2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -1723.9946289, 3273.3566895, -1728.6478271, 3281.9042969, -5005.8989258, 5002.0043945
1: -582.9691162, 1230.9879150, -584.5093994, 1234.2640381, -1817.2331543, 1815.4973145
2: -298.5708618, 1239.3637695, -299.3667603, 1242.6334229, -1541.2042236, 1538.7304688
3: -680.9635010, 1508.8701172, -682.7730713, 1512.8596191, -2193.8232422, 2191.6430664
4: -387.5766907, 1275.2492676, -388.6107178, 1278.6130371, -1666.1896973, 1663.8599854

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_A1_B1_A1

### Relational analysis result of NS_A2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1899648, upper bound: 3656.1879425
time: 0.75 seconds

## Relational analysis of NS_A2_A2_A1_B1_A2

### Relational analysis result of NS_A2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1899902, upper bound: 3656.1894059
time: 0.83 seconds

## BFS NS instance: NS_A2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -1723.9946289, 3273.3566895, -1686.1866455, 3198.5917969, -4922.5859375, 4959.5434570
1: -582.9691162, 1230.9879150, -569.4422607, 1202.2066650, -1785.1757812, 1800.4301758
2: -298.5708618, 1239.3637695, -291.7250366, 1210.6175537, -1509.1882324, 1531.0888672
3: -680.9635010, 1508.8701172, -665.1435547, 1473.6713867, -2154.6347656, 2174.0136719
4: -387.5766907, 1275.2492676, -378.6471252, 1245.7604980, -1633.3371582, 1653.8963623

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_A1_B2_A1

### Relational analysis result of NS_A2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1899648, upper bound: 3656.1879425
time: 0.86 seconds

## Relational analysis of NS_A2_A2_A1_B2_A2

### Relational analysis result of NS_A2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1899902, upper bound: 3656.1894059
time: 0.68 seconds

## BFS NS instance: NS_A2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -1681.5596924, 3190.1123047, -1728.6478271, 3281.9042969, -4963.4638672, 4918.7602539
1: -567.9144287, 1198.9532471, -584.5093994, 1234.2640381, -1802.1784668, 1783.4624023
2: -290.9331665, 1207.3736572, -299.3667603, 1242.6334229, -1533.5666504, 1506.7404785
3: -663.3439941, 1469.7086182, -682.7730713, 1512.8596191, -2176.2036133, 2152.4814453
4: -377.6191406, 1242.4111328, -388.6107178, 1278.6130371, -1656.2321777, 1631.0218506

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_A2_B1_A1

### Relational analysis result of NS_A2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1865417, upper bound: 3656.1868379
time: 0.73 seconds

## Relational analysis of NS_A2_A2_A2_B1_A2

### Relational analysis result of NS_A2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1894059, upper bound: 3656.1894488
time: 0.76 seconds

## BFS NS instance: NS_A2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -1681.5596924, 3190.1123047, -1686.1866455, 3198.5917969, -4880.1508789, 4876.2988281
1: -567.9144287, 1198.9532471, -569.4422607, 1202.2066650, -1770.1210938, 1768.3955078
2: -290.9331665, 1207.3736572, -291.7250366, 1210.6175537, -1501.5507812, 1499.0986328
3: -663.3439941, 1469.7086182, -665.1435547, 1473.6713867, -2137.0153809, 2134.8520508
4: -377.6191406, 1242.4111328, -378.6471252, 1245.7604980, -1623.3795166, 1621.0582275

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_A2_B2_A1

### Relational analysis result of NS_A2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1865417, upper bound: 3656.1868379
time: 0.90 seconds

## Relational analysis of NS_A2_A2_A2_B2_A2

### Relational analysis result of NS_A2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1894059, upper bound: 3656.1894488
time: 0.79 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.81 seconds
NS_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -3656.1612855, upper bound: 3656.1615208
NS_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -3656.1612855, upper bound: 3656.1622517
NS_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -3656.1612855, upper bound: 3656.1615208
NS_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -3656.1612855, upper bound: 3656.1622517
NS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -3656.1612432, upper bound: 3656.1612432
NS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -3656.1612432, upper bound: 3656.1612432
NS_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -3656.1612432, upper bound: 3656.1666350
NS_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -3656.1612432, upper bound: 3656.1622591
NS_A2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -3656.1899648, upper bound: 3656.1879425
NS_A2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -3656.1899902, upper bound: 3656.1894059
NS_A2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -3656.1899648, upper bound: 3656.1879425
NS_A2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -3656.1899902, upper bound: 3656.1894059
NS_A2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -3656.1865417, upper bound: 3656.1868379
NS_A2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -3656.1894059, upper bound: 3656.1894488
NS_A2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -3656.1865417, upper bound: 3656.1868379
NS_A2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -3656.1894059, upper bound: 3656.1894488

## BFS NS instance: NS_A2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1528.8487549, 2936.8110352, -1529.7244873, 2938.4519043, -4467.3007812, 4466.5341797
1: -520.2947388, 1098.7904053, -520.5869141, 1099.4128418, -1619.7075195, 1619.3773193
2: -265.7568970, 1107.6166992, -265.9141541, 1108.2401123, -1373.9968262, 1373.5308838
3: -607.3670044, 1347.3883057, -607.7248535, 1348.1492920, -1955.5163574, 1955.1129150
4: -344.7153320, 1141.4797363, -344.9177856, 1142.1367188, -1486.8519287, 1486.3973389

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## BFS NS instance: NS_A2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1528.8487549, 2936.8110352, -1592.6188965, 3041.4145508, -4570.2617188, 4529.4296875
1: -520.2947388, 1098.7904053, -539.8632202, 1139.6384277, -1659.9331055, 1638.6535645
2: -265.7568970, 1107.6166992, -276.0022583, 1148.5240479, -1414.2808838, 1383.6188965
3: -607.3670044, 1347.3883057, -629.9686890, 1397.2849121, -2004.6518555, 1977.3569336
4: -344.7153320, 1141.4797363, -357.9005432, 1182.2995605, -1527.0146484, 1499.3802490

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 23

## BFS NS instance: NS_A2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1676.4122314, 3195.7221680, -1529.7244873, 2938.4519043, -4614.8642578, 4725.4467773
1: -568.0551147, 1200.6782227, -520.5869141, 1099.4128418, -1667.4680176, 1721.2651367
2: -290.8211365, 1209.1031494, -265.9141541, 1108.2401123, -1399.0610352, 1475.0173340
3: -664.2241821, 1472.2062988, -607.7248535, 1348.1492920, -2012.3735352, 2079.9306641
4: -377.6401672, 1244.4934082, -344.9177856, 1142.1367188, -1519.7768555, 1589.4111328

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 11

## BFS NS instance: NS_A2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1676.4122314, 3195.7221680, -1592.6188965, 3041.4145508, -4717.8261719, 4788.3408203
1: -568.0551147, 1200.6782227, -539.8632202, 1139.6384277, -1707.6936035, 1740.5415039
2: -290.8211365, 1209.1031494, -276.0022583, 1148.5240479, -1439.3450928, 1485.1054688
3: -664.2241821, 1472.2062988, -629.9686890, 1397.2849121, -2061.5090332, 2102.1750488
4: -377.6401672, 1244.4934082, -357.9005432, 1182.2995605, -1559.9396973, 1602.3939209

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 23

## BFS NS instance: NS_A2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -1528.8487549, 2936.8110352, -1677.6610107, 3198.0446777, -4726.8925781, 4614.4716797
1: -520.2947388, 1098.7904053, -568.4644165, 1201.5491943, -1721.8439941, 1667.2548828
2: -265.7568970, 1107.6166992, -291.0398560, 1209.9788818, -1475.7358398, 1398.6564941
3: -607.3670044, 1347.3883057, -664.7223511, 1473.2760010, -2080.6430664, 2012.1105957
4: -344.7153320, 1141.4797363, -377.9221802, 1245.4234619, -1590.1386719, 1519.4018555

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -1676.4122314, 3195.7221680, -1677.6610107, 3198.0446777, -4874.4565430, 4873.3833008
1: -568.0551147, 1200.6782227, -568.4644165, 1201.5491943, -1769.6042480, 1769.1425781
2: -290.8211365, 1209.1031494, -291.0398560, 1209.9788818, -1500.8000488, 1500.1430664
3: -664.2241821, 1472.2062988, -664.7223511, 1473.2760010, -2137.5000000, 2136.9282227
4: -377.6401672, 1244.4934082, -377.9221802, 1245.4234619, -1623.0635986, 1622.4155273

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## BFS NS instance: NS_A2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -1528.8487549, 2936.8110352, -1740.6629639, 3304.0895996, -4832.9384766, 4677.4736328
1: -520.2947388, 1098.7904053, -588.3560791, 1242.5281982, -1762.8229980, 1687.1464844
2: -265.7568970, 1107.6166992, -301.3850403, 1251.0102539, -1516.7670898, 1409.0014648
3: -607.3670044, 1347.3883057, -687.2623901, 1523.1033936, -2130.4704590, 2034.6506348
4: -344.7153320, 1141.4797363, -391.1883850, 1287.2303467, -1631.9455566, 1532.6680908

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -1676.4122314, 3195.7221680, -1740.6629639, 3304.0895996, -4980.5019531, 4936.3852539
1: -568.0551147, 1200.6782227, -588.3560791, 1242.5281982, -1810.5832520, 1789.0343018
2: -290.8211365, 1209.1031494, -301.3850403, 1251.0102539, -1541.8314209, 1510.4880371
3: -664.2241821, 1472.2062988, -687.2623901, 1523.1033936, -2187.3271484, 2159.4687500
4: -377.6401672, 1244.4934082, -391.1883850, 1287.2303467, -1664.8704834, 1635.6817627

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 11

## BFS NS instance: NS_A2_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1667.0397949, 3159.8005371, -1691.4049072, 3207.4538574, -4874.4926758, 4851.2055664
1: -563.2828979, 1189.5279541, -571.6372070, 1207.0964355, -1770.3793945, 1761.1647949
2: -288.6612549, 1197.3933105, -292.8986206, 1215.1035156, -1503.7646484, 1490.2919922
3: -658.2575073, 1457.6895752, -667.9476318, 1479.3416748, -2137.5991211, 2125.6367188
4: -374.7251282, 1232.5289307, -380.2175293, 1250.6627197, -1625.3878174, 1612.7464600

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_A1_B1_A1_B1

### Relational analysis result of NS_A2_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1888190, upper bound: 3656.1888190
time: 0.81 seconds

## Relational analysis of NS_A2_A2_A1_B1_A1_B2

### Relational analysis result of NS_A2_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1888190, upper bound: 3656.1888502
time: 0.76 seconds

## BFS NS instance: NS_A2_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1761.2929688, 3332.3117676, -1637.2595215, 3116.6030273, -4877.8959961, 4969.5712891
1: -594.5371094, 1251.5583496, -554.4498291, 1170.0604248, -1764.5975342, 1806.0081787
2: -304.6340027, 1261.3537598, -283.9993896, 1179.1575928, -1483.7912598, 1545.3531494
3: -692.1199951, 1532.6373291, -647.3817749, 1434.9921875, -2127.1120605, 2180.0190430
4: -394.9266052, 1298.0366211, -368.5002441, 1213.9434814, -1608.8699951, 1666.5368652

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_A1_B1_A2_B1

### Relational analysis result of NS_A2_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1888502, upper bound: 3656.1899698
time: 0.80 seconds

## Relational analysis of NS_A2_A2_A1_B1_A2_B2

### Relational analysis result of NS_A2_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1888502, upper bound: 3656.1899902
time: 0.84 seconds

## BFS NS instance: NS_A2_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1667.0397949, 3159.8005371, -1649.0999756, 3124.6230469, -4791.6625977, 4808.9003906
1: -563.2828979, 1189.5279541, -556.6049194, 1175.0732422, -1738.3562012, 1746.1328125
2: -288.6612549, 1197.3933105, -285.2685242, 1183.1307373, -1471.7918701, 1482.6616211
3: -658.2575073, 1457.6895752, -650.3391724, 1440.1914062, -2098.4489746, 2108.0283203
4: -374.7251282, 1232.5289307, -370.2606812, 1217.8587646, -1592.5838623, 1602.7895508

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_A1_B2_A1_B1

### Relational analysis result of NS_A2_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1873452, upper bound: 3656.1859834
time: 0.72 seconds

## Relational analysis of NS_A2_A2_A1_B2_A1_B2

### Relational analysis result of NS_A2_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1873452, upper bound: 3656.1879425
time: 0.92 seconds

## BFS NS instance: NS_A2_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1761.2929688, 3332.3117676, -1607.0841064, 3054.9914551, -4816.2841797, 4939.3959961
1: -594.5371094, 1251.5583496, -543.4451904, 1146.7296143, -1741.2667236, 1795.0032959
2: -304.6340027, 1261.3537598, -278.4231873, 1155.6177979, -1460.2517090, 1539.7769775
3: -692.1199951, 1532.6373291, -634.5120239, 1406.3098145, -2098.4296875, 2167.1494141
4: -394.9266052, 1298.0366211, -361.2492065, 1189.6453857, -1584.5720215, 1659.2858887

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_A1_B2_A2_B1

### Relational analysis result of NS_A2_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1873972, upper bound: 3656.1865417
time: 0.82 seconds

## Relational analysis of NS_A2_A2_A1_B2_A2_B2

### Relational analysis result of NS_A2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1873972, upper bound: 3656.1894059
time: 0.79 seconds

## BFS NS instance: NS_A2_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1624.9265137, 3077.0727539, -1691.4049072, 3207.4538574, -4832.3793945, 4768.4775391
1: -548.2611084, 1157.5198975, -571.6372070, 1207.0964355, -1755.3575439, 1729.1568604
2: -281.0467224, 1165.4528809, -292.8986206, 1215.1035156, -1496.1501465, 1458.3515625
3: -640.6859131, 1418.5307617, -667.9476318, 1479.3416748, -2120.0275879, 2086.4777832
4: -364.7857361, 1199.7465820, -380.2175293, 1250.6627197, -1615.4484863, 1579.9639893

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_A2_B1_A1_B1

### Relational analysis result of NS_A2_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1859834, upper bound: 3656.1873452
time: 0.75 seconds

## Relational analysis of NS_A2_A2_A2_B1_A1_B2

### Relational analysis result of NS_A2_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1859834, upper bound: 3656.1873972
time: 0.74 seconds

## BFS NS instance: NS_A2_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1738.4870605, 3286.9797363, -1637.2595215, 3116.6030273, -4855.0898438, 4924.2387695
1: -586.3719482, 1234.6223145, -554.4498291, 1170.0604248, -1756.4323730, 1789.0721436
2: -300.5553284, 1244.2427979, -283.9993896, 1179.1575928, -1479.7128906, 1528.2421875
3: -682.9289551, 1511.8227539, -647.3817749, 1434.9921875, -2117.9211426, 2159.2045898
4: -389.6261902, 1280.2789307, -368.5002441, 1213.9434814, -1603.5693359, 1648.7791748

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_A2_B1_A2_B1

### Relational analysis result of NS_A2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1879425, upper bound: 3656.1899785
time: 0.88 seconds

## Relational analysis of NS_A2_A2_A2_B1_A2_B2

### Relational analysis result of NS_A2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1879425, upper bound: 3656.1900120
time: 0.82 seconds

## BFS NS instance: NS_A2_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1624.9265137, 3077.0727539, -1649.0999756, 3124.6230469, -4749.5493164, 4726.1728516
1: -548.2611084, 1157.5198975, -556.6049194, 1175.0732422, -1723.3343506, 1714.1247559
2: -281.0467224, 1165.4528809, -285.2685242, 1183.1307373, -1464.1774902, 1450.7210693
3: -640.6859131, 1418.5307617, -650.3391724, 1440.1914062, -2080.8774414, 2068.8693848
4: -364.7857361, 1199.7465820, -370.2606812, 1217.8587646, -1582.6444092, 1570.0073242

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_A2_B2_A1_B1

### Relational analysis result of NS_A2_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1854581, upper bound: 3656.1854581
time: 0.75 seconds

## Relational analysis of NS_A2_A2_A2_B2_A1_B2

### Relational analysis result of NS_A2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1854581, upper bound: 3656.1868379
time: 0.86 seconds

## BFS NS instance: NS_A2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1738.4870605, 3286.9797363, -1607.0841064, 3054.9914551, -4793.4780273, 4894.0639648
1: -586.3719482, 1234.6223145, -543.4451904, 1146.7296143, -1733.1015625, 1778.0673828
2: -300.5553284, 1244.2427979, -278.4231873, 1155.6177979, -1456.1730957, 1522.6658936
3: -682.9289551, 1511.8227539, -634.5120239, 1406.3098145, -2089.2387695, 2146.3347168
4: -389.6261902, 1280.2789307, -361.2492065, 1189.6453857, -1579.2713623, 1641.5280762

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_A2_B2_A2_B1

### Relational analysis result of NS_A2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1868379, upper bound: 3656.1865750
time: 0.78 seconds

## Relational analysis of NS_A2_A2_A2_B2_A2_B2

### Relational analysis result of NS_A2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1868379, upper bound: 3656.1894488
time: 0.73 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.65 seconds
NS_A2_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -3656.1888190, upper bound: 3656.1888190
NS_A2_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -3656.1888190, upper bound: 3656.1888502
NS_A2_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -3656.1888502, upper bound: 3656.1899698
NS_A2_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -3656.1888502, upper bound: 3656.1899902
NS_A2_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -3656.1873452, upper bound: 3656.1859834
NS_A2_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -3656.1873452, upper bound: 3656.1879425
NS_A2_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -3656.1873972, upper bound: 3656.1865417
NS_A2_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -3656.1873972, upper bound: 3656.1894059
NS_A2_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -3656.1859834, upper bound: 3656.1873452
NS_A2_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -3656.1859834, upper bound: 3656.1873972
NS_A2_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -3656.1879425, upper bound: 3656.1899785
NS_A2_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -3656.1879425, upper bound: 3656.1900120
NS_A2_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -3656.1854581, upper bound: 3656.1854581
NS_A2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -3656.1854581, upper bound: 3656.1868379
NS_A2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -3656.1868379, upper bound: 3656.1865750
NS_A2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 0, lower bound: -3656.1868379, upper bound: 3656.1894488

## BFS NS instance: NS_A2_A2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1667.0397949, 3159.8005371, -1671.6643066, 3168.3137207, -4835.3535156, 4831.4648438
1: -563.2828979, 1189.5279541, -564.8126221, 1192.7908936, -1756.0737305, 1754.3404541
2: -288.6612549, 1197.3933105, -289.4537964, 1200.6552734, -1489.3164062, 1486.8469238
3: -658.2575073, 1457.6895752, -660.0601196, 1461.6660156, -2119.9235840, 2117.7495117
4: -374.7251282, 1232.5289307, -375.7544250, 1235.8851318, -1610.6102295, 1608.2833252

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_A1_B1_A1_B1_B1

### Relational analysis result of NS_A2_A2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1779124, upper bound: 3656.1768219
time: 0.78 seconds

## Relational analysis of NS_A2_A2_A1_B1_A1_B1_B2

### Relational analysis result of NS_A2_A2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1559344, upper bound: 3656.1630742
time: 0.76 seconds

## BFS NS instance: NS_A2_A2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1667.0397949, 3159.8005371, -1766.3977051, 3341.8259277, -5008.8652344, 4926.1982422
1: -563.2828979, 1189.5279541, -596.2349854, 1255.1651611, -1818.4479980, 1785.7629395
2: -288.6612549, 1197.3933105, -305.5222168, 1264.9880371, -1553.6491699, 1502.9155273
3: -658.2575073, 1457.6895752, -694.1506348, 1537.0639648, -2195.3215332, 2151.8395996
4: -374.7251282, 1232.5289307, -396.0743713, 1301.8270264, -1676.5521240, 1628.6032715

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1

### Relational analysis result of NS_A2_A2_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1779124, upper bound: 3656.1768219
time: 0.77 seconds

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2

### Relational analysis result of NS_A2_A2_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1559344, upper bound: 3656.1630742
time: 0.80 seconds

## BFS NS instance: NS_A2_A2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1761.2929688, 3332.3117676, -1671.6643066, 3168.3137207, -4929.6064453, 5003.9760742
1: -594.5371094, 1251.5583496, -564.8126221, 1192.7908936, -1787.3280029, 1816.3709717
2: -304.6340027, 1261.3537598, -289.4537964, 1200.6552734, -1505.2890625, 1550.8073730
3: -692.1199951, 1532.6373291, -660.0601196, 1461.6660156, -2153.7861328, 2192.6975098
4: -394.9266052, 1298.0366211, -375.7544250, 1235.8851318, -1630.8117676, 1673.7910156

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_A2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1565410, upper bound: 3656.1677452
time: 1.07 seconds

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_A2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -3656.1523610, upper bound: 3656.1523610
time: 0.74 seconds

## BFS NS instance: NS_A2_A2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1761.2929688, 3332.3117676, -1766.4573975, 3341.9375000, -5103.2299805, 5098.7690430
1: -594.5371094, 1251.5583496, -596.2543335, 1255.2056885, -1849.7427979, 1847.8126221
2: -304.6340027, 1261.3537598, -305.5321960, 1265.0289307, -1569.6627197, 1566.8858643
3: -692.1199951, 1532.6373291, -694.1726074, 1537.1143799, -2229.2336426, 2226.8098145
4: -394.9266052, 1298.0366211, -396.0870972, 1301.8690186, -1696.7956543, 1694.1237793

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A1_B1_A2_B2_B1

### Relational analysis result of NS_A2_A2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1863010, upper bound: 3656.1847990
time: 0.74 seconds

## Relational analysis of NS_A2_A2_A1_B1_A2_B2_B2

### Relational analysis result of NS_A2_A2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1888502, upper bound: 3656.1899902
time: 1.44 seconds

## BFS NS instance: NS_A2_A2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1667.0397949, 3159.8005371, -1629.5869141, 3085.6640625, -4752.7041016, 4789.3876953
1: -563.2828979, 1189.5279541, -549.8078613, 1160.8277588, -1724.1105957, 1739.3355713
2: -288.6612549, 1197.3933105, -281.8485413, 1168.7565918, -1457.4176025, 1479.2418213
3: -658.2575073, 1457.6895752, -642.5105591, 1422.5587158, -2080.8161621, 2100.1997070
4: -374.7251282, 1232.5289307, -365.8284912, 1203.1389160, -1577.8640137, 1598.3574219

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A1_B2_A1_B1_B1

### Relational analysis result of NS_A2_A2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1858434, upper bound: 3656.1841173
time: 0.70 seconds

## Relational analysis of NS_A2_A2_A1_B2_A1_B1_B2

### Relational analysis result of NS_A2_A2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1873452, upper bound: 3656.1859942
time: 0.77 seconds

## BFS NS instance: NS_A2_A2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1667.0397949, 3159.8005371, -1742.9810791, 3295.3210449, -4962.3608398, 4902.7817383
1: -563.2828979, 1189.5279541, -587.8641357, 1237.7784424, -1801.0612793, 1777.3920898
2: -288.6612549, 1197.3933105, -301.3349609, 1247.4266357, -1536.0876465, 1498.7282715
3: -658.2575073, 1457.6895752, -684.7066650, 1515.6986084, -2173.9560547, 2142.3959961
4: -374.7251282, 1232.5289307, -390.6321716, 1283.6018066, -1658.3269043, 1623.1610107

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A1_B2_A1_B2_B1

### Relational analysis result of NS_A2_A2_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1858434, upper bound: 3656.1841173
time: 0.70 seconds

## Relational analysis of NS_A2_A2_A1_B2_A1_B2_B2

### Relational analysis result of NS_A2_A2_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1873452, upper bound: 3656.1879425
time: 0.81 seconds

## BFS NS instance: NS_A2_A2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1761.2929688, 3332.3117676, -1629.5869141, 3085.6640625, -4846.9570312, 4961.8979492
1: -594.5371094, 1251.5583496, -549.8078613, 1160.8277588, -1755.3648682, 1801.3659668
2: -304.6340027, 1261.3537598, -281.8485413, 1168.7565918, -1473.3903809, 1543.2022705
3: -692.1199951, 1532.6373291, -642.5105591, 1422.5587158, -2114.6784668, 2175.1479492
4: -394.9266052, 1298.0366211, -365.8284912, 1203.1389160, -1598.0655518, 1663.8651123

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_A2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1837642, upper bound: 3656.1853217
time: 0.73 seconds

## Relational analysis of NS_A2_A2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_A2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1873972, upper bound: 3656.1865417
time: 0.72 seconds

## BFS NS instance: NS_A2_A2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1761.2929688, 3332.3117676, -1743.2463379, 3295.8146973, -5057.1074219, 5075.5581055
1: -594.5371094, 1251.5583496, -587.9487305, 1237.9600830, -1832.4971924, 1839.5069580
2: -304.6340027, 1261.3537598, -301.3789673, 1247.6101074, -1552.2438965, 1562.7326660
3: -692.1199951, 1532.6373291, -684.8048096, 1515.9213867, -2208.0410156, 2217.4421387
4: -394.9266052, 1298.0366211, -390.6888733, 1283.7885742, -1678.7152100, 1688.7254639

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A1_B2_A2_B2_B1

### Relational analysis result of NS_A2_A2_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1859998, upper bound: 3656.1845845
time: 0.74 seconds

## Relational analysis of NS_A2_A2_A1_B2_A2_B2_B2

### Relational analysis result of NS_A2_A2_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1873972, upper bound: 3656.1894059
time: 0.76 seconds

## BFS NS instance: NS_A2_A2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1624.9265137, 3077.0727539, -1671.6643066, 3168.3137207, -4793.2402344, 4748.7373047
1: -548.2611084, 1157.5198975, -564.8126221, 1192.7908936, -1741.0520020, 1722.3325195
2: -281.0467224, 1165.4528809, -289.4537964, 1200.6552734, -1481.7019043, 1454.9063721
3: -640.6859131, 1418.5307617, -660.0601196, 1461.6660156, -2102.3520508, 2078.5905762
4: -364.7857361, 1199.7465820, -375.7544250, 1235.8851318, -1600.6707764, 1575.5007324

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_A2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1849826, upper bound: 3656.1866417
time: 0.77 seconds

## Relational analysis of NS_A2_A2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_A2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1859746, upper bound: 3656.1873452
time: 0.75 seconds

## BFS NS instance: NS_A2_A2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1624.9265137, 3077.0727539, -1766.3977051, 3341.8259277, -4966.7514648, 4843.4707031
1: -548.2611084, 1157.5198975, -596.2349854, 1255.1651611, -1803.4262695, 1753.7548828
2: -281.0467224, 1165.4528809, -305.5222168, 1264.9880371, -1546.0346680, 1470.9748535
3: -640.6859131, 1418.5307617, -694.1506348, 1537.0639648, -2177.7500000, 2112.6806641
4: -364.7857361, 1199.7465820, -396.0743713, 1301.8270264, -1666.6126709, 1595.8209229

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A2_B1_A1_B2_B1

### Relational analysis result of NS_A2_A2_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1853217, upper bound: 3656.1837642
time: 0.79 seconds

## Relational analysis of NS_A2_A2_A2_B1_A1_B2_B2

### Relational analysis result of NS_A2_A2_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1859747, upper bound: 3656.1873972
time: 0.92 seconds

## BFS NS instance: NS_A2_A2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1738.4870605, 3286.9797363, -1671.6643066, 3168.3137207, -4906.8007812, 4958.6440430
1: -586.3719482, 1234.6223145, -564.8126221, 1192.7908936, -1779.1628418, 1799.4349365
2: -300.5553284, 1244.2427979, -289.4537964, 1200.6552734, -1501.2105713, 1533.6961670
3: -682.9289551, 1511.8227539, -660.0601196, 1461.6660156, -2144.5949707, 2171.8828125
4: -389.6261902, 1280.2789307, -375.7544250, 1235.8851318, -1625.5111084, 1656.0333252

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_A2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1840884, upper bound: 3656.1858434
time: 0.77 seconds

## Relational analysis of NS_A2_A2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_A2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1879425, upper bound: 3656.1899785
time: 0.78 seconds

## BFS NS instance: NS_A2_A2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1738.4870605, 3286.9797363, -1766.4573975, 3341.9375000, -5080.4243164, 5053.4370117
1: -586.3719482, 1234.6223145, -596.2543335, 1255.2056885, -1841.5775146, 1830.8764648
2: -300.5553284, 1244.2427979, -305.5321960, 1265.0289307, -1565.5842285, 1549.7747803
3: -682.9289551, 1511.8227539, -694.1726074, 1537.1143799, -2220.0429688, 2205.9951172
4: -389.6261902, 1280.2789307, -396.0870972, 1301.8690186, -1691.4951172, 1676.3659668

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_A2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1840884, upper bound: 3656.1858434
time: 0.79 seconds

## Relational analysis of NS_A2_A2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_A2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1879425, upper bound: 3656.1900120
time: 0.76 seconds

## BFS NS instance: NS_A2_A2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1624.9265137, 3077.0727539, -1629.5869141, 3085.6640625, -4710.5908203, 4706.6596680
1: -548.2611084, 1157.5198975, -549.8078613, 1160.8277588, -1709.0888672, 1707.3275146
2: -281.0467224, 1165.4528809, -281.8485413, 1168.7565918, -1449.8032227, 1447.3012695
3: -640.6859131, 1418.5307617, -642.5105591, 1422.5587158, -2063.2443848, 2061.0407715
4: -364.7857361, 1199.7465820, -365.8284912, 1203.1389160, -1567.9245605, 1565.5749512

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A2_B2_A1_B1_B1

### Relational analysis result of NS_A2_A2_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1850518, upper bound: 3656.1835513
time: 0.80 seconds

## Relational analysis of NS_A2_A2_A2_B2_A1_B1_B2

### Relational analysis result of NS_A2_A2_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1854460, upper bound: 3656.1854460
time: 0.77 seconds

## BFS NS instance: NS_A2_A2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1624.9265137, 3077.0727539, -1742.9810791, 3295.3210449, -4920.2475586, 4820.0537109
1: -548.2611084, 1157.5198975, -587.8641357, 1237.7784424, -1786.0395508, 1745.3839111
2: -281.0467224, 1165.4528809, -301.3349609, 1247.4266357, -1528.4732666, 1466.7877197
3: -640.6859131, 1418.5307617, -684.7066650, 1515.6986084, -2156.3845215, 2103.2370605
4: -364.7857361, 1199.7465820, -390.6321716, 1283.6018066, -1648.3874512, 1590.3784180

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_A2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_A2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1850518, upper bound: 3656.1835513
time: 0.72 seconds

## Relational analysis of NS_A2_A2_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_A2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1854460, upper bound: 3656.1868229
time: 0.72 seconds

## BFS NS instance: NS_A2_A2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1738.4870605, 3286.9797363, -1629.5869141, 3085.6640625, -4824.1513672, 4916.5659180
1: -586.3719482, 1234.6223145, -549.8078613, 1160.8277588, -1747.1995850, 1784.4300537
2: -300.5553284, 1244.2427979, -281.8485413, 1168.7565918, -1469.3118896, 1526.0913086
3: -682.9289551, 1511.8227539, -642.5105591, 1422.5587158, -2105.4875488, 2154.3332520
4: -389.6261902, 1280.2789307, -365.8284912, 1203.1389160, -1592.7650146, 1646.1074219

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_A2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1835513, upper bound: 3656.1850530
time: 1.00 seconds

## Relational analysis of NS_A2_A2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_A2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1868229, upper bound: 3656.1865750
time: 0.95 seconds

## BFS NS instance: NS_A2_A2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1738.4870605, 3286.9797363, -1743.2463379, 3295.8146973, -5034.3012695, 5030.2260742
1: -586.3719482, 1234.6223145, -587.9487305, 1237.9600830, -1824.3320312, 1822.5708008
2: -300.5553284, 1244.2427979, -301.3789673, 1247.6101074, -1548.1654053, 1545.6217041
3: -682.9289551, 1511.8227539, -684.8048096, 1515.9213867, -2198.8503418, 2196.6274414
4: -389.6261902, 1280.2789307, -390.6888733, 1283.7885742, -1673.4145508, 1670.9677734

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A2_B2_A2_B2_B1

### Relational analysis result of NS_A2_A2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1857947, upper bound: 3656.1845972
time: 0.80 seconds

## Relational analysis of NS_A2_A2_A2_B2_A2_B2_B2

### Relational analysis result of NS_A2_A2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1868229, upper bound: 3656.1894488
time: 0.81 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.23 seconds
NS_A2_A2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1779124, upper bound: 3656.1768219
NS_A2_A2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1559344, upper bound: 3656.1630742
NS_A2_A2_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1779124, upper bound: 3656.1768219
NS_A2_A2_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1559344, upper bound: 3656.1630742
NS_A2_A2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1565410, upper bound: 3656.1677452
NS_A2_A2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1523610, upper bound: 3656.1523610
NS_A2_A2_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1863010, upper bound: 3656.1847990
NS_A2_A2_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1888502, upper bound: 3656.1899902
NS_A2_A2_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1858434, upper bound: 3656.1841173
NS_A2_A2_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1873452, upper bound: 3656.1859942
NS_A2_A2_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1858434, upper bound: 3656.1841173
NS_A2_A2_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1873452, upper bound: 3656.1879425
NS_A2_A2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1837642, upper bound: 3656.1853217
NS_A2_A2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1873972, upper bound: 3656.1865417
NS_A2_A2_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1859998, upper bound: 3656.1845845
NS_A2_A2_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1873972, upper bound: 3656.1894059
NS_A2_A2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1849826, upper bound: 3656.1866417
NS_A2_A2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1859746, upper bound: 3656.1873452
NS_A2_A2_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1853217, upper bound: 3656.1837642
NS_A2_A2_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1859747, upper bound: 3656.1873972
NS_A2_A2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1840884, upper bound: 3656.1858434
NS_A2_A2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1879425, upper bound: 3656.1899785
NS_A2_A2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1840884, upper bound: 3656.1858434
NS_A2_A2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1879425, upper bound: 3656.1900120
NS_A2_A2_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1850518, upper bound: 3656.1835513
NS_A2_A2_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1854460, upper bound: 3656.1854460
NS_A2_A2_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1850518, upper bound: 3656.1835513
NS_A2_A2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1854460, upper bound: 3656.1868229
NS_A2_A2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1835513, upper bound: 3656.1850530
NS_A2_A2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1868229, upper bound: 3656.1865750
NS_A2_A2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1857947, upper bound: 3656.1845972
NS_A2_A2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.23
Output dim: 0, lower bound: -3656.1868229, upper bound: 3656.1894488

## BFS NS instance: NS_A2_A2_A1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -1539.6311035, 2920.3776855, -1507.4084473, 2838.8740234, -4378.5034180, 4427.7856445
1: -520.3101807, 1099.1741943, -507.0930176, 1070.5950928, -1590.9052734, 1606.2670898
2: -266.2427979, 1106.2229004, -259.1033020, 1076.3062744, -1342.5490723, 1365.3258057
3: -607.1586914, 1346.1905518, -590.8200073, 1309.2779541, -1916.4366455, 1937.0103760
4: -345.7644348, 1137.1783447, -336.6302185, 1105.3138428, -1451.0782471, 1473.8084717

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_A1_B1_A1_B1_B1_A1

### Relational analysis result of NS_A2_A2_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1757645, upper bound: 3656.1757645
time: 0.87 seconds

## Relational analysis of NS_A2_A2_A1_B1_A1_B1_B1_A2

### Relational analysis result of NS_A2_A2_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1757645, upper bound: 3656.1787186
time: 0.82 seconds

## BFS NS instance: NS_A2_A2_A1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -1606.7797852, 3046.1293945, -1580.1934814, 2996.2873535, -4603.0664062, 4626.3227539
1: -543.2612305, 1146.8358154, -534.4556885, 1128.0761719, -1671.3374023, 1681.2910156
2: -278.2746887, 1154.2947998, -273.7331238, 1135.4343262, -1413.7088623, 1428.0279541
3: -633.7748413, 1405.0859375, -623.1156006, 1382.0784912, -2015.8532715, 2028.2015381
4: -361.2109375, 1186.8266602, -355.2994690, 1167.0230713, -1528.2338867, 1542.1260986

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_A1_B1_A1_B1_B2_A1

### Relational analysis result of NS_A2_A2_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1787186, upper bound: 3656.1757646
time: 0.85 seconds

## Relational analysis of NS_A2_A2_A1_B1_A1_B1_B2_A2

### Relational analysis result of NS_A2_A2_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1787186, upper bound: 3656.1801932
time: 0.77 seconds

## BFS NS instance: NS_A2_A2_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -1539.6311035, 2920.3776855, -1678.6152344, 3153.0671387, -4692.6977539, 4598.9931641
1: -520.3101807, 1099.1741943, -563.6798706, 1186.6053467, -1706.9155273, 1662.8540039
2: -266.2427979, 1106.2229004, -287.8667908, 1194.5675049, -1460.8103027, 1394.0894775
3: -607.1586914, 1346.1905518, -653.9186401, 1450.8355713, -2057.9941406, 2000.1091309
4: -345.7644348, 1137.1783447, -373.3535461, 1225.4223633, -1571.1867676, 1510.5318604

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A1

### Relational analysis result of NS_A2_A2_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1559344, upper bound: 3656.1630742
time: 0.77 seconds

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A2

### Relational analysis result of NS_A2_A2_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1559344, upper bound: 3656.1630742
time: 0.78 seconds

## BFS NS instance: NS_A2_A2_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -1606.7797852, 3046.1293945, -1674.7963867, 3169.8242188, -4776.6030273, 4720.9248047
1: -543.2612305, 1146.8358154, -565.7226562, 1190.1917725, -1733.4530029, 1712.5584717
2: -278.2746887, 1154.2947998, -289.9687805, 1199.7789307, -1478.0535889, 1444.2635498
3: -633.7748413, 1405.0859375, -658.4281006, 1457.5803223, -2091.3552246, 2063.5141602
4: -361.2109375, 1186.8266602, -375.8463745, 1234.7199707, -1595.9305420, 1562.6730957

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A1

### Relational analysis result of NS_A2_A2_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1559344, upper bound: 3656.1630742
time: 0.71 seconds

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A2

### Relational analysis result of NS_A2_A2_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1559344, upper bound: 3656.1630742
time: 0.80 seconds

## BFS NS instance: NS_A2_A2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1673.4432373, 3143.4572754, -1543.9197998, 2928.2111816, -4601.6538086, 4687.3754883
1: -561.9579468, 1182.9489746, -521.7227783, 1102.1917725, -1664.1496582, 1704.6717529
2: -286.9740906, 1190.8913574, -266.9759827, 1109.2333984, -1396.2073975, 1457.8673096
3: -651.8730469, 1446.3576660, -608.8379517, 1349.8751221, -2001.7479248, 2055.1955566
4: -372.1984558, 1221.5997314, -346.7171326, 1140.2805176, -1512.4790039, 1568.3168945

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_A2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1630742, upper bound: 3656.1559344
time: 0.73 seconds

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_A2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1630742, upper bound: 3656.1559344
time: 0.86 seconds

## BFS NS instance: NS_A2_A2_A1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -1716.0737305, 3247.3041992, -1669.5976562, 3165.8557129, -4881.9296875, 4916.9018555
1: -579.7223511, 1220.6928711, -563.9788818, 1187.9143066, -1767.6364746, 1784.6717529
2: -297.0895386, 1229.9635010, -289.1115112, 1196.3835449, -1493.4729004, 1519.0749512
3: -674.9565430, 1494.7678223, -657.6522827, 1455.2485352, -2130.2043457, 2152.4199219
4: -385.2847290, 1265.4114990, -375.0876770, 1231.4923096, -1616.7769775, 1640.4991455

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A1_B1_A2_B2_B1_A1

### Relational analysis result of NS_A2_A2_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1836852, upper bound: 3656.1836852
time: 0.81 seconds

## Relational analysis of NS_A2_A2_A1_B1_A2_B2_B1_A2

### Relational analysis result of NS_A2_A2_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1836852, upper bound: 3656.1847990
time: 0.74 seconds

## BFS NS instance: NS_A2_A2_A1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -1752.2227783, 3314.2343750, -1752.2880859, 3313.7009277, -5065.9238281, 5066.5224609
1: -591.4793091, 1244.9252930, -591.4755249, 1244.8388672, -1836.3181152, 1836.4006348
2: -303.0088806, 1254.6677246, -302.9928284, 1254.5814209, -1557.5902100, 1557.6605225
3: -688.0986938, 1524.4644775, -687.8952026, 1524.3458252, -2212.4443359, 2212.3596191
4: -392.7975464, 1290.7991943, -392.7601318, 1290.5717773, -1683.3692627, 1683.5592041

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A1_B1_A2_B2_B2_A1

### Relational analysis result of NS_A2_A2_A1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1843073, upper bound: 3656.1861459
time: 0.80 seconds

## Relational analysis of NS_A2_A2_A1_B1_A2_B2_B2_A2

### Relational analysis result of NS_A2_A2_A1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1843073, upper bound: 3656.1899902
time: 0.84 seconds

## BFS NS instance: NS_A2_A2_A1_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -1613.0932617, 3057.8427734, -1570.6818848, 2968.7041016, -4581.7963867, 4628.5244141
1: -545.5236206, 1152.2304688, -528.4856567, 1115.7889404, -1661.3125000, 1680.7159424
2: -279.4571228, 1159.1783447, -270.8729553, 1123.8266602, -1403.2835693, 1430.0511475
3: -636.8363037, 1411.4532471, -617.3443604, 1367.0949707, -2003.9312744, 2028.7976074
4: -362.9771423, 1192.0927734, -351.4655151, 1156.7523193, -1519.7294922, 1543.5583496

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A1_B2_A1_B1_B1_A1

### Relational analysis result of NS_A2_A2_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1863299, upper bound: 3656.1850216
time: 0.80 seconds

## Relational analysis of NS_A2_A2_A1_B2_A1_B1_B1_A2

### Relational analysis result of NS_A2_A2_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1863299, upper bound: 3656.1850216
time: 0.77 seconds

## BFS NS instance: NS_A2_A2_A1_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -1654.3358154, 3135.2810059, -1615.3469238, 3058.1804199, -4712.5161133, 4750.6274414
1: -559.0637817, 1180.5465088, -545.0556641, 1150.7160645, -1709.7797852, 1725.6021729
2: -286.4484558, 1188.3282471, -279.3717957, 1158.5865479, -1445.0347900, 1467.6999512
3: -653.1193237, 1446.5950928, -636.8099976, 1410.1042480, -2063.2233887, 2083.4050293
4: -371.8651428, 1223.0406494, -362.6220093, 1192.5836182, -1564.4486084, 1585.6625977

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A1_B2_A1_B1_B2_A1

### Relational analysis result of NS_A2_A2_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1869801, upper bound: 3656.1859942
time: 0.80 seconds

## Relational analysis of NS_A2_A2_A1_B2_A1_B1_B2_A2

### Relational analysis result of NS_A2_A2_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1869801, upper bound: 3656.1859942
time: 0.75 seconds

## BFS NS instance: NS_A2_A2_A1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -1613.0932617, 3057.8427734, -1648.2941895, 3121.4946289, -4734.5874023, 4706.1367188
1: -545.5236206, 1152.2304688, -555.8372803, 1171.3210449, -1716.8444824, 1708.0675049
2: -279.4571228, 1159.1783447, -285.1421509, 1179.4586182, -1458.9155273, 1444.3201904
3: -636.8363037, 1411.4532471, -648.6629028, 1434.8021240, -2071.6384277, 2060.1157227
4: -362.9771423, 1192.0927734, -369.9239502, 1213.9868164, -1576.9639893, 1562.0167236

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A1_B2_A1_B2_B1_A1

### Relational analysis result of NS_A2_A2_A1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1855874, upper bound: 3656.1841173
time: 0.75 seconds

## Relational analysis of NS_A2_A2_A1_B2_A1_B2_B1_A2

### Relational analysis result of NS_A2_A2_A1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1855874, upper bound: 3656.1841173
time: 1.04 seconds

## BFS NS instance: NS_A2_A2_A1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -1654.3358154, 3135.2810059, -1731.0451660, 3271.4824219, -4925.8173828, 4866.3261719
1: -559.0637817, 1180.5465088, -583.8411865, 1229.0253906, -1788.0891113, 1764.3876953
2: -286.4484558, 1188.3282471, -299.2206726, 1238.6389160, -1525.0871582, 1487.5488281
3: -653.1193237, 1446.5950928, -679.4330444, 1504.9721680, -2158.0910645, 2126.0280762
4: -371.8651428, 1223.0406494, -387.8525391, 1274.1148682, -1645.9797363, 1610.8931885

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A1_B2_A1_B2_B2_A1

### Relational analysis result of NS_A2_A2_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1883951, upper bound: 3656.1869728
time: 0.94 seconds

## Relational analysis of NS_A2_A2_A1_B2_A1_B2_B2_A2

### Relational analysis result of NS_A2_A2_A1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1883951, upper bound: 3656.1879425
time: 0.83 seconds

## BFS NS instance: NS_A2_A2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1665.6662598, 3158.5205078, -1577.0043945, 2986.0124512, -4651.6782227, 4735.5234375
1: -562.6630249, 1185.1271973, -532.4622803, 1124.6917725, -1687.3547363, 1717.5894775
2: -288.4227905, 1193.5793457, -272.9234314, 1131.6870117, -1420.1097412, 1466.5028076
3: -656.0850830, 1451.8333740, -621.7258301, 1377.6823730, -2033.7673340, 2073.5590820
4: -374.1973572, 1228.5595703, -354.4539795, 1163.8529053, -1538.0502930, 1583.0135498

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_A2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1836047, upper bound: 3656.1844003
time: 0.77 seconds

## Relational analysis of NS_A2_A2_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_A2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1836047, upper bound: 3656.1853217
time: 0.79 seconds

## BFS NS instance: NS_A2_A2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1747.0953369, 3304.0253906, -1619.6864014, 3066.3906250, -4813.4858398, 4923.7119141
1: -589.7482300, 1241.1733398, -546.4871826, 1153.7629395, -1743.5112305, 1787.6605225
2: -302.0896301, 1250.8876953, -280.1228333, 1161.6490479, -1463.7382812, 1531.0103760
3: -685.8314209, 1519.8452148, -638.5282593, 1413.8554688, -2099.6865234, 2158.3730469
4: -391.5929565, 1286.7207031, -363.5937805, 1195.7708740, -1587.3637695, 1650.3144531

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_A2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1866602, upper bound: 3656.1854725
time: 0.76 seconds

## Relational analysis of NS_A2_A2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_A2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1866602, upper bound: 3656.1865417
time: 0.87 seconds

## BFS NS instance: NS_A2_A2_A1_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -1716.0737305, 3247.3041992, -1648.8555908, 3122.5458984, -4838.6191406, 4896.1596680
1: -579.7223511, 1220.6928711, -556.0182495, 1171.7005615, -1751.4228516, 1776.7110596
2: -297.0895386, 1229.9635010, -285.2345886, 1179.8453369, -1476.9346924, 1515.1981201
3: -674.9565430, 1494.7678223, -648.8707275, 1435.2716064, -2110.2275391, 2143.6384277
4: -385.2847290, 1265.4114990, -370.0421753, 1214.3808594, -1599.6655273, 1635.4536133

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A1_B2_A2_B2_B1_A1

### Relational analysis result of NS_A2_A2_A1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1834202, upper bound: 3656.1834945
time: 0.75 seconds

## Relational analysis of NS_A2_A2_A1_B2_A2_B2_B1_A2

### Relational analysis result of NS_A2_A2_A1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1834202, upper bound: 3656.1845845
time: 0.86 seconds

## BFS NS instance: NS_A2_A2_A1_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -1752.2227783, 3314.2343750, -1731.2995605, 3271.9560547, -5024.1787109, 5045.5341797
1: -591.4793091, 1244.9252930, -583.9221802, 1229.1995850, -1820.6788330, 1828.8474121
2: -303.0088806, 1254.6677246, -299.2627563, 1238.8146973, -1541.8233643, 1553.9304199
3: -688.0986938, 1524.4644775, -679.5265503, 1505.1854248, -2193.2841797, 2203.9909668
4: -392.7975464, 1290.7991943, -387.9068604, 1274.2927246, -1667.0902100, 1678.7060547

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A1_B2_A2_B2_B2_A1

### Relational analysis result of NS_A2_A2_A1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1837642, upper bound: 3656.1855445
time: 0.76 seconds

## Relational analysis of NS_A2_A2_A1_B2_A2_B2_B2_A2

### Relational analysis result of NS_A2_A2_A1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1837642, upper bound: 3656.1894059
time: 0.83 seconds

## BFS NS instance: NS_A2_A2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1567.8541260, 2963.4162598, -1617.8363037, 3066.5993652, -4634.4521484, 4581.2524414
1: -527.5475464, 1113.8009033, -547.0965576, 1155.5841064, -1683.1315918, 1660.8974609
2: -270.3831482, 1121.8249512, -280.2723999, 1162.5343018, -1432.9174805, 1402.0971680
3: -616.2122803, 1364.6499023, -638.6834106, 1415.5428467, -2031.7551270, 2003.3332520
4: -350.8308411, 1154.6561279, -364.0365295, 1195.5496826, -1546.3804932, 1518.6926270

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_A2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1850216, upper bound: 3656.1863299
time: 0.77 seconds

## Relational analysis of NS_A2_A2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_A2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1850216, upper bound: 3656.1866417
time: 0.78 seconds

## BFS NS instance: NS_A2_A2_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1610.7145996, 3049.6674805, -1658.9510498, 3143.7656250, -4754.4794922, 4708.6186523
1: -543.5228271, 1147.4403076, -560.5891724, 1183.8031006, -1727.3258057, 1708.0292969
2: -278.5771790, 1155.3122559, -287.2385254, 1191.5848389, -1470.1618652, 1442.5507812
3: -635.0007935, 1406.1126709, -654.9173584, 1450.5655518, -2085.5664062, 2061.0300293
4: -361.5896912, 1189.2198486, -372.8907776, 1226.3912354, -1587.9809570, 1562.1105957

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_A2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1859942, upper bound: 3656.1869801
time: 0.78 seconds

## Relational analysis of NS_A2_A2_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_A2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1859942, upper bound: 3656.1873452
time: 0.85 seconds

## BFS NS instance: NS_A2_A2_A2_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -1572.1572266, 2977.0102539, -1669.2912598, 3165.2836914, -4737.4404297, 4646.3017578
1: -530.8500977, 1121.2515869, -563.8806152, 1187.7071533, -1718.5572510, 1685.1322021
2: -272.0871887, 1128.2404785, -289.0609741, 1196.1727295, -1468.2597656, 1417.3015137
3: -619.8245850, 1373.4892578, -657.5390015, 1454.9926758, -2074.8173828, 2031.0279541
4: -353.3672180, 1160.3010254, -375.0230713, 1231.2783203, -1584.6455078, 1535.3240967

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A2_B1_A1_B2_B1_A1

### Relational analysis result of NS_A2_A2_A2_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1844003, upper bound: 3656.1836047
time: 0.76 seconds

## Relational analysis of NS_A2_A2_A2_B1_A1_B2_B1_A2

### Relational analysis result of NS_A2_A2_A2_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1844003, upper bound: 3656.1837642
time: 0.78 seconds

## BFS NS instance: NS_A2_A2_A2_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -1615.0142822, 3057.7893066, -1752.2310791, 3313.5947266, -4928.6083984, 4810.0205078
1: -544.9392090, 1150.4555664, -591.4573364, 1244.7996826, -1789.7388916, 1741.9128418
2: -279.3199768, 1158.3442383, -302.9833069, 1254.5418701, -1533.8618164, 1461.3273926
3: -636.7016602, 1409.8271484, -687.8742065, 1524.2977295, -2160.9995117, 2097.7011719
4: -362.5502930, 1192.3754883, -392.7478333, 1290.5316162, -1653.0819092, 1585.1232910

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A2_B1_A1_B2_B2_A1

### Relational analysis result of NS_A2_A2_A2_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1854725, upper bound: 3656.1866602
time: 0.76 seconds

## Relational analysis of NS_A2_A2_A2_B1_A1_B2_B2_A2

### Relational analysis result of NS_A2_A2_A2_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1854725, upper bound: 3656.1873972
time: 0.90 seconds

## BFS NS instance: NS_A2_A2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1644.9135742, 3115.2236328, -1617.8363037, 3066.5993652, -4711.5117188, 4733.0595703
1: -554.7155151, 1168.9741211, -547.0965576, 1155.5841064, -1710.2995605, 1716.0706787
2: -284.5486145, 1177.0771484, -280.2723999, 1162.5343018, -1447.0828857, 1457.3496094
3: -647.3126831, 1431.9102783, -638.6834106, 1415.5428467, -2062.8549805, 2070.5935059
4: -369.1610413, 1211.4613037, -364.0365295, 1195.5496826, -1564.7106934, 1575.4978027

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_A2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1841173, upper bound: 3656.1855874
time: 0.92 seconds

## Relational analysis of NS_A2_A2_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_A2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1841173, upper bound: 3656.1858434
time: 0.80 seconds

## BFS NS instance: NS_A2_A2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1726.5937500, 3263.2263184, -1658.9510498, 3143.7656250, -4870.3593750, 4922.1772461
1: -582.3646240, 1225.9025879, -560.5891724, 1183.8031006, -1766.1677246, 1786.4915771
2: -298.4488220, 1235.4880371, -287.2385254, 1191.5848389, -1490.0336914, 1522.7265625
3: -677.6727905, 1501.1368408, -654.9173584, 1450.5655518, -2128.2382812, 2156.0541992
4: -386.8565674, 1270.8245850, -372.8907776, 1226.3912354, -1613.2478027, 1643.7153320

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_A2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1869727, upper bound: 3656.1883951
time: 0.81 seconds

## Relational analysis of NS_A2_A2_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_A2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1869727, upper bound: 3656.1899785
time: 0.78 seconds

## BFS NS instance: NS_A2_A2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1644.9135742, 3115.2236328, -1721.2401123, 3256.9680176, -4901.8813477, 4836.4628906
1: -554.7155151, 1168.9741211, -581.4462891, 1224.3446045, -1779.0599365, 1750.4204102
2: -284.5486145, 1177.0771484, -297.9901123, 1233.6436768, -1518.1922607, 1475.0671387
3: -647.3126831, 1431.9102783, -677.0125122, 1499.2535400, -2146.5659180, 2108.9228516
4: -369.1610413, 1211.4613037, -386.4473877, 1269.2517090, -1638.4127197, 1597.9086914

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_A2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1834945, upper bound: 3656.1834511
time: 0.69 seconds

## Relational analysis of NS_A2_A2_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_A2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1834945, upper bound: 3656.1858434
time: 0.75 seconds

## BFS NS instance: NS_A2_A2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1726.5937500, 3263.2263184, -1757.3912354, 3323.8681641, -5050.4614258, 5020.6166992
1: -582.3646240, 1225.9025879, -593.1975098, 1248.5743408, -1830.9388428, 1819.1000977
2: -298.4488220, 1235.4880371, -303.9076843, 1258.3449707, -1556.7938232, 1539.3955078
3: -677.6727905, 1501.1368408, -690.1528320, 1528.9445801, -2206.6171875, 2191.2895508
4: -386.8565674, 1270.8245850, -393.9590759, 1294.6335449, -1681.4901123, 1664.7836914

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_A2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1860940, upper bound: 3656.1848051
time: 0.81 seconds

## Relational analysis of NS_A2_A2_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_A2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1860940, upper bound: 3656.1900120
time: 0.84 seconds

## BFS NS instance: NS_A2_A2_A2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -1572.1572266, 2977.0102539, -1570.6818848, 2968.7041016, -4540.8603516, 4547.6923828
1: -530.8500977, 1121.2515869, -528.4856567, 1115.7889404, -1646.6390381, 1649.7373047
2: -272.0871887, 1128.2404785, -270.8729553, 1123.8266602, -1395.9138184, 1399.1134033
3: -619.8245850, 1373.4892578, -617.3443604, 1367.0949707, -1986.9195557, 1990.8334961
4: -353.3672180, 1160.3010254, -351.4655151, 1156.7523193, -1510.1195068, 1511.7666016

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A2_B2_A1_B1_B1_A1

### Relational analysis result of NS_A2_A2_A2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1843286, upper bound: 3656.1843286
time: 0.71 seconds

## Relational analysis of NS_A2_A2_A2_B2_A1_B1_B1_A2

### Relational analysis result of NS_A2_A2_A2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1843286, upper bound: 3656.1844630
time: 0.77 seconds

## BFS NS instance: NS_A2_A2_A2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -1615.0142822, 3057.7893066, -1615.3469238, 3058.1804199, -4673.1948242, 4673.1362305
1: -544.9392090, 1150.4555664, -545.0556641, 1150.7160645, -1695.6552734, 1695.5112305
2: -279.3199768, 1158.3442383, -279.3717957, 1158.5865479, -1437.9062500, 1437.7158203
3: -636.7016602, 1409.8271484, -636.8099976, 1410.1042480, -2046.8059082, 2046.6369629
4: -362.5502930, 1192.3754883, -362.6220093, 1192.5836182, -1555.1339111, 1554.9975586

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A2_B2_A1_B1_B2_A1

### Relational analysis result of NS_A2_A2_A2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1844630, upper bound: 3656.1852683
time: 0.82 seconds

## Relational analysis of NS_A2_A2_A2_B2_A1_B1_B2_A2

### Relational analysis result of NS_A2_A2_A2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1844630, upper bound: 3656.1854460
time: 0.77 seconds

## BFS NS instance: NS_A2_A2_A2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -1572.1572266, 2977.0102539, -1648.2941895, 3121.4946289, -4693.6513672, 4625.3046875
1: -530.8500977, 1121.2515869, -555.8372803, 1171.3210449, -1702.1711426, 1677.0888672
2: -272.0871887, 1128.2404785, -285.1421509, 1179.4586182, -1451.5456543, 1413.3825684
3: -619.8245850, 1373.4892578, -648.6629028, 1434.8021240, -2054.6267090, 2022.1517334
4: -353.3672180, 1160.3010254, -369.9239502, 1213.9868164, -1567.3540039, 1530.2249756

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A2_B2_A1_B2_B1_A1

### Relational analysis result of NS_A2_A2_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1841720, upper bound: 3656.1834129
time: 0.73 seconds

## Relational analysis of NS_A2_A2_A2_B2_A1_B2_B1_A2

### Relational analysis result of NS_A2_A2_A2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1841720, upper bound: 3656.1835513
time: 0.75 seconds

## BFS NS instance: NS_A2_A2_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -1615.0142822, 3057.7893066, -1731.0451660, 3271.4824219, -4886.4951172, 4788.8344727
1: -544.9392090, 1150.4555664, -583.8411865, 1229.0253906, -1773.9645996, 1734.2967529
2: -279.3199768, 1158.3442383, -299.2206726, 1238.6389160, -1517.9586182, 1457.5646973
3: -636.7016602, 1409.8271484, -679.4330444, 1504.9721680, -2141.6738281, 2089.2602539
4: -362.5502930, 1192.3754883, -387.8525391, 1274.1148682, -1636.6651611, 1580.2280273

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A2_B2_A1_B2_B2_A1

### Relational analysis result of NS_A2_A2_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1854725, upper bound: 3656.1861229
time: 0.75 seconds

## Relational analysis of NS_A2_A2_A2_B2_A1_B2_B2_A2

### Relational analysis result of NS_A2_A2_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1854725, upper bound: 3656.1868229
time: 0.78 seconds

## BFS NS instance: NS_A2_A2_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1644.9135742, 3115.2236328, -1577.0043945, 2986.0124512, -4630.9248047, 4692.2270508
1: -554.7155151, 1168.9741211, -532.4622803, 1124.6917725, -1679.4072266, 1701.4364014
2: -284.5486145, 1177.0771484, -272.9234314, 1131.6870117, -1416.2354736, 1450.0006104
3: -647.3126831, 1431.9102783, -621.7258301, 1377.6823730, -2024.9949951, 2053.6362305
4: -369.1610413, 1211.4613037, -354.4539795, 1163.8529053, -1533.0139160, 1565.9152832

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_A2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1834129, upper bound: 3656.1841720
time: 0.79 seconds

## Relational analysis of NS_A2_A2_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_A2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1834129, upper bound: 3656.1850530
time: 0.83 seconds

## BFS NS instance: NS_A2_A2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1726.5937500, 3263.2263184, -1619.6864014, 3066.3906250, -4792.9838867, 4882.9121094
1: -582.3646240, 1225.9025879, -546.4871826, 1153.7629395, -1736.1271973, 1772.3897705
2: -298.4488220, 1235.4880371, -280.1228333, 1161.6490479, -1460.0977783, 1515.6107178
3: -677.6727905, 1501.1368408, -638.5282593, 1413.8554688, -2091.5278320, 2139.6650391
4: -386.8565674, 1270.8245850, -363.5937805, 1195.7708740, -1582.6274414, 1634.4183350

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_A2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1861228, upper bound: 3656.1854858
time: 0.79 seconds

## Relational analysis of NS_A2_A2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_A2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1861228, upper bound: 3656.1865750
time: 0.84 seconds

## BFS NS instance: NS_A2_A2_A2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -1696.5367432, 3207.1469727, -1648.8555908, 3122.5458984, -4819.0825195, 4856.0024414
1: -572.4740601, 1205.7912598, -556.0182495, 1171.7005615, -1744.1745605, 1761.8094482
2: -293.4895020, 1214.8280029, -285.2345886, 1179.8453369, -1473.3347168, 1500.0626221
3: -666.7922363, 1476.3764648, -648.8707275, 1435.2716064, -2102.0639648, 2125.2465820
4: -380.5877991, 1249.6064453, -370.0421753, 1214.3808594, -1594.9686279, 1619.6486816

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A2_B2_A2_B2_B1_A1

### Relational analysis result of NS_A2_A2_A2_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1832332, upper bound: 3656.1832671
time: 0.75 seconds

## Relational analysis of NS_A2_A2_A2_B2_A2_B2_B1_A2

### Relational analysis result of NS_A2_A2_A2_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1832332, upper bound: 3656.1845972
time: 0.79 seconds

## BFS NS instance: NS_A2_A2_A2_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -1730.1762695, 3270.4321289, -1731.2995605, 3271.9560547, -5002.1323242, 5001.7314453
1: -583.5712280, 1228.5328369, -583.9221802, 1229.1995850, -1812.7707520, 1812.4550781
2: -299.0874939, 1238.1329346, -299.2627563, 1238.8146973, -1537.9022217, 1537.3955078
3: -679.2891846, 1504.3659668, -679.5265503, 1505.1854248, -2184.4746094, 2183.8925781
4: -387.6986389, 1273.7126465, -387.9068604, 1274.2927246, -1661.9913330, 1661.6195068

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A2_B2_A2_B2_B2_A1

### Relational analysis result of NS_A2_A2_A2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1835513, upper bound: 3656.1853642
time: 0.93 seconds

## Relational analysis of NS_A2_A2_A2_B2_A2_B2_B2_A2

### Relational analysis result of NS_A2_A2_A2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1835513, upper bound: 3656.1894488
time: 0.86 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 4.11 seconds
NS_A2_A2_A1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1757645, upper bound: 3656.1757645
NS_A2_A2_A1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1757645, upper bound: 3656.1787186
NS_A2_A2_A1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1787186, upper bound: 3656.1757646
NS_A2_A2_A1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1787186, upper bound: 3656.1801932
NS_A2_A2_A1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1559344, upper bound: 3656.1630742
NS_A2_A2_A1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1559344, upper bound: 3656.1630742
NS_A2_A2_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1559344, upper bound: 3656.1630742
NS_A2_A2_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1559344, upper bound: 3656.1630742
NS_A2_A2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1630742, upper bound: 3656.1559344
NS_A2_A2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1630742, upper bound: 3656.1559344
NS_A2_A2_A1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1836852, upper bound: 3656.1836852
NS_A2_A2_A1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1836852, upper bound: 3656.1847990
NS_A2_A2_A1_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1843073, upper bound: 3656.1861459
NS_A2_A2_A1_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1843073, upper bound: 3656.1899902
NS_A2_A2_A1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1863299, upper bound: 3656.1850216
NS_A2_A2_A1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1863299, upper bound: 3656.1850216
NS_A2_A2_A1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1869801, upper bound: 3656.1859942
NS_A2_A2_A1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1869801, upper bound: 3656.1859942
NS_A2_A2_A1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1855874, upper bound: 3656.1841173
NS_A2_A2_A1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1855874, upper bound: 3656.1841173
NS_A2_A2_A1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1883951, upper bound: 3656.1869728
NS_A2_A2_A1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1883951, upper bound: 3656.1879425
NS_A2_A2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1836047, upper bound: 3656.1844003
NS_A2_A2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1836047, upper bound: 3656.1853217
NS_A2_A2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1866602, upper bound: 3656.1854725
NS_A2_A2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1866602, upper bound: 3656.1865417
NS_A2_A2_A1_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1834202, upper bound: 3656.1834945
NS_A2_A2_A1_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1834202, upper bound: 3656.1845845
NS_A2_A2_A1_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1837642, upper bound: 3656.1855445
NS_A2_A2_A1_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1837642, upper bound: 3656.1894059
NS_A2_A2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1850216, upper bound: 3656.1863299
NS_A2_A2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1850216, upper bound: 3656.1866417
NS_A2_A2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1859942, upper bound: 3656.1869801
NS_A2_A2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1859942, upper bound: 3656.1873452
NS_A2_A2_A2_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1844003, upper bound: 3656.1836047
NS_A2_A2_A2_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1844003, upper bound: 3656.1837642
NS_A2_A2_A2_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1854725, upper bound: 3656.1866602
NS_A2_A2_A2_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1854725, upper bound: 3656.1873972
NS_A2_A2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1841173, upper bound: 3656.1855874
NS_A2_A2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1841173, upper bound: 3656.1858434
NS_A2_A2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1869727, upper bound: 3656.1883951
NS_A2_A2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1869727, upper bound: 3656.1899785
NS_A2_A2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1834945, upper bound: 3656.1834511
NS_A2_A2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1834945, upper bound: 3656.1858434
NS_A2_A2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1860940, upper bound: 3656.1848051
NS_A2_A2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1860940, upper bound: 3656.1900120
NS_A2_A2_A2_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1843286, upper bound: 3656.1843286
NS_A2_A2_A2_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1843286, upper bound: 3656.1844630
NS_A2_A2_A2_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1844630, upper bound: 3656.1852683
NS_A2_A2_A2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1844630, upper bound: 3656.1854460
NS_A2_A2_A2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1841720, upper bound: 3656.1834129
NS_A2_A2_A2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1841720, upper bound: 3656.1835513
NS_A2_A2_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1854725, upper bound: 3656.1861229
NS_A2_A2_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1854725, upper bound: 3656.1868229
NS_A2_A2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1834129, upper bound: 3656.1841720
NS_A2_A2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1834129, upper bound: 3656.1850530
NS_A2_A2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1861228, upper bound: 3656.1854858
NS_A2_A2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1861228, upper bound: 3656.1865750
NS_A2_A2_A2_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1832332, upper bound: 3656.1832671
NS_A2_A2_A2_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1832332, upper bound: 3656.1845972
NS_A2_A2_A2_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1835513, upper bound: 3656.1853642
NS_A2_A2_A2_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 4.11
Output dim: 0, lower bound: -3656.1835513, upper bound: 3656.1894488

## BFS NS instance: NS_A2_A2_A1_B1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -1503.2408447, 2831.1301270, -1507.4084473, 2838.8740234, -4342.1132812, 4338.5380859
1: -505.7030334, 1067.6181641, -507.0930176, 1070.5950928, -1576.2980957, 1574.7111816
2: -258.3850708, 1073.3292236, -259.1033020, 1076.3062744, -1334.6910400, 1332.4323730
3: -589.1818237, 1305.6480713, -590.8200073, 1309.2779541, -1898.4597168, 1896.4677734
4: -335.6979675, 1102.2519531, -336.6302185, 1105.3138428, -1441.0118408, 1438.8822021

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_A1_B1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_A1_B1_A1_B1_B1_A1_B1

### Relational analysis result of NS_A2_A2_A1_B1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1739850, upper bound: 3656.1726076
time: 0.78 seconds

## Relational analysis of NS_A2_A2_A1_B1_A1_B1_B1_A1_B2

### Relational analysis result of NS_A2_A2_A1_B1_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1774094, upper bound: 3656.1750861
time: 0.83 seconds

## BFS NS instance: NS_A2_A2_A1_B1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -1571.8886719, 2980.9716797, -1507.4084473, 2838.8740234, -4410.7617188, 4488.3798828
1: -531.7463379, 1122.3138428, -507.0930176, 1070.5950928, -1602.3414307, 1629.4068604
2: -272.2844849, 1129.6293945, -259.1033020, 1076.3062744, -1348.5908203, 1388.7322998
3: -619.7531738, 1374.9692383, -590.8200073, 1309.2779541, -1929.0311279, 1965.7889404
4: -353.4244385, 1160.8282471, -336.6302185, 1105.3138428, -1458.7382812, 1497.4584961

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_A2_A1_B1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_A1_B1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A1_B1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_A1_B1_A1_B1_B1_A2_B1

### Relational analysis result of NS_A2_A2_A1_B1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1739850, upper bound: 3656.1760661
time: 0.73 seconds

## Relational analysis of NS_A2_A2_A1_B1_A1_B1_B1_A2_B2

### Relational analysis result of NS_A2_A2_A1_B1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1774094, upper bound: 3656.1750861
time: 0.83 seconds

## BFS NS instance: NS_A2_A2_A1_B1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -1503.2408447, 2831.1301270, -1580.1934814, 2996.2873535, -4499.5273438, 4411.3237305
1: -505.7030334, 1067.6181641, -534.4556885, 1128.0761719, -1633.7791748, 1602.0736084
2: -258.3850708, 1073.3292236, -273.7331238, 1135.4343262, -1393.8189697, 1347.0623779
3: -589.1818237, 1305.6480713, -623.1156006, 1382.0784912, -1971.2602539, 1928.7635498
4: -335.6979675, 1102.2519531, -355.2994690, 1167.0230713, -1502.7210693, 1457.5513916

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_A2_A1_B1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_A1_B1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A1_B1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_A1_B1_A1_B1_B2_A1_A1

### Relational analysis result of NS_A2_A2_A1_B1_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1726076, upper bound: 3656.1736673
time: 0.81 seconds

## Relational analysis of NS_A2_A2_A1_B1_A1_B1_B2_A1_A2

### Relational analysis result of NS_A2_A2_A1_B1_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1747917, upper bound: 3656.1747967
time: 0.80 seconds

## BFS NS instance: NS_A2_A2_A1_B1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -1575.5386963, 2987.6970215, -1580.1934814, 2996.2873535, -4571.8261719, 4567.8906250
1: -532.9169312, 1124.8094482, -534.4556885, 1128.0761719, -1660.9927979, 1659.2647705
2: -272.9318848, 1132.1525879, -273.7331238, 1135.4343262, -1408.3658447, 1405.8857422
3: -621.2743530, 1378.0787354, -623.1156006, 1382.0784912, -2003.3527832, 2001.1943359
4: -354.2606506, 1163.5922852, -355.2994690, 1167.0230713, -1521.2836914, 1518.8917236

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_A2_A1_B1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_A1_B1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A1_B1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_A1_B1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A1_B1_A1_B1_B2_A2_B1

### Relational analysis result of NS_A2_A2_A1_B1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1755684, upper bound: 3656.1776916
time: 0.92 seconds

## Relational analysis of NS_A2_A2_A1_B1_A1_B1_B2_A2_B2

### Relational analysis result of NS_A2_A2_A1_B1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1757464, upper bound: 3656.1777404
time: 0.83 seconds

## BFS NS instance: NS_A2_A2_A1_B1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -1503.2408447, 2831.1301270, -1678.6152344, 3153.0671387, -4656.3076172, 4509.7451172
1: -505.7030334, 1067.6181641, -563.6798706, 1186.6053467, -1692.3083496, 1631.2980957
2: -258.3850708, 1073.3292236, -287.8667908, 1194.5675049, -1452.9522705, 1361.1960449
3: -589.1818237, 1305.6480713, -653.9186401, 1450.8355713, -2040.0173340, 1959.5666504
4: -335.6979675, 1102.2519531, -373.3535461, 1225.4223633, -1561.1203613, 1475.6054688

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A1_B1

### Relational analysis result of NS_A2_A2_A1_B1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1807680, upper bound: 3656.1760065
time: 0.73 seconds

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A1_A1

### Relational analysis result of NS_A2_A2_A1_B1_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1807668, upper bound: 3656.1747960
time: 0.79 seconds

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A1_A1

### Relational analysis result of NS_A2_A2_A1_B1_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1793358, upper bound: 3656.1747697
time: 0.80 seconds

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A1_A2

### Relational analysis result of NS_A2_A2_A1_B1_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1807680, upper bound: 3656.1760065
time: 2.63 seconds

## BFS NS instance: NS_A2_A2_A1_B1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -1571.8886719, 2980.9716797, -1678.6152344, 3153.0671387, -4724.9560547, 4659.5869141
1: -531.7463379, 1122.3138428, -563.6798706, 1186.6053467, -1718.3516846, 1685.9936523
2: -272.2844849, 1129.6293945, -287.8667908, 1194.5675049, -1466.8520508, 1417.4959717
3: -619.7531738, 1374.9692383, -653.9186401, 1450.8355713, -2070.5888672, 2028.8879395
4: -353.4244385, 1160.8282471, -373.3535461, 1225.4223633, -1578.8468018, 1534.1817627

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A2_A1

### Relational analysis result of NS_A2_A2_A1_B1_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1807668, upper bound: 3656.1747960
time: 0.74 seconds

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A2_B1

### Relational analysis result of NS_A2_A2_A1_B1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1807680, upper bound: 3656.1768219
time: 0.82 seconds

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A2_B1

### Relational analysis result of NS_A2_A2_A1_B1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1807540, upper bound: 3656.1764260
time: 0.75 seconds

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B1_A2_B2

### Relational analysis result of NS_A2_A2_A1_B1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1807680, upper bound: 3656.1768219
time: 0.76 seconds

## BFS NS instance: NS_A2_A2_A1_B1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -1503.2408447, 2831.1301270, -1674.7963867, 3169.8242188, -4673.0644531, 4505.9262695
1: -505.7030334, 1067.6181641, -565.7226562, 1190.1917725, -1695.8947754, 1633.3408203
2: -258.3850708, 1073.3292236, -289.9687805, 1199.7789307, -1458.1639404, 1363.2979736
3: -589.1818237, 1305.6480713, -658.4281006, 1457.5803223, -2046.7622070, 1964.0761719
4: -335.6979675, 1102.2519531, -375.8463745, 1234.7199707, -1570.4179688, 1478.0983887

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A1_B1

### Relational analysis result of NS_A2_A2_A1_B1_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1558405, upper bound: 3656.1630742
time: 0.74 seconds

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A1_B1

### Relational analysis result of NS_A2_A2_A1_B1_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3656.1316346, upper bound: 3656.1457365
time: 0.89 seconds

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A1_B2

### Relational analysis result of NS_A2_A2_A1_B1_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1524205, upper bound: 3656.1608509
time: 0.75 seconds

## BFS NS instance: NS_A2_A2_A1_B1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -1575.5386963, 2987.6970215, -1674.7963867, 3169.8242188, -4745.3627930, 4662.4931641
1: -532.9169312, 1124.8094482, -565.7226562, 1190.1917725, -1723.1086426, 1690.5321045
2: -272.9318848, 1132.1525879, -289.9687805, 1199.7789307, -1472.7106934, 1422.1213379
3: -621.2743530, 1378.0787354, -658.4281006, 1457.5803223, -2078.8547363, 2036.5068359
4: -354.2606506, 1163.5922852, -375.8463745, 1234.7199707, -1588.9805908, 1539.4387207

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 48

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A2_B1

### Relational analysis result of NS_A2_A2_A1_B1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1558405, upper bound: 3656.1630742
time: 0.78 seconds

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A2_B1

### Relational analysis result of NS_A2_A2_A1_B1_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -3656.1316346, upper bound: 3656.1461897
time: 1.02 seconds

## Relational analysis of NS_A2_A2_A1_B1_A1_B2_B2_A2_B2

### Relational analysis result of NS_A2_A2_A1_B1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1524205, upper bound: 3656.1608509
time: 0.95 seconds

## BFS NS instance: NS_A2_A2_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1673.4432373, 3143.4572754, -1507.4084473, 2838.8740234, -4512.3154297, 4650.8642578
1: -561.9579468, 1182.9489746, -507.0930176, 1070.5950928, -1632.5529785, 1690.0419922
2: -286.9740906, 1190.8913574, -259.1033020, 1076.3062744, -1363.2802734, 1449.9943848
3: -651.8730469, 1446.3576660, -590.8200073, 1309.2779541, -1961.1510010, 2037.1774902
4: -372.1984558, 1221.5997314, -336.6302185, 1105.3138428, -1477.5123291, 1558.2299805

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_A2_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1760065, upper bound: 3656.1807680
time: 0.78 seconds

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B1_B1

### Relational analysis result of NS_A2_A2_A1_B1_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1747959, upper bound: 3656.1807668
time: 1.19 seconds

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_A2_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1760065, upper bound: 3656.1807680
time: 0.75 seconds

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_A2_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1759256, upper bound: 3656.1795647
time: 0.81 seconds

## BFS NS instance: NS_A2_A2_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1673.4432373, 3143.4572754, -1576.4615479, 2989.4204102, -4662.8632812, 4719.9179688
1: -561.9579468, 1182.9489746, -533.2595825, 1125.5272217, -1687.4851074, 1716.2084961
2: -286.9740906, 1190.8913574, -273.0724792, 1132.8572998, -1419.8312988, 1463.9636230
3: -651.8730469, 1446.3576660, -621.5643921, 1378.9041748, -2030.7770996, 2067.9218750
4: -372.1984558, 1221.5997314, -354.4461060, 1164.2032471, -1536.4017334, 1576.0458984

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B2_B1

### Relational analysis result of NS_A2_A2_A1_B1_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1747959, upper bound: 3656.1807668
time: 0.75 seconds

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_A2_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1760065, upper bound: 3656.1807680
time: 0.76 seconds

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_A2_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1758544, upper bound: 3656.1807540
time: 0.76 seconds

## Relational analysis of NS_A2_A2_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_A2_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3656.1760065, upper bound: 3656.1807680
time: 0.74 seconds

## BFS NS instance: NS_A2_A2_A1_B1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -1665.6662598, 3158.5205078, -1669.5976562, 3165.8557129, -4831.5219727, 4828.1181641
1: -562.6630249, 1185.1271973, -563.9788818, 1187.9143066, -1750.5770264, 1749.1060791
2: -288.4227905, 1193.5793457, -289.1115112, 1196.3835449, -1484.8062744, 1482.6907959
3: -656.0850830, 1451.8333740, -657.6522827, 1455.2485352, -2111.3332520, 2109.4853516
4: -374.1973572, 1228.5595703, -375.0876770, 1231.4923096, -1605.6896973, 1603.6472168

Time for backsubstitution: 2.09 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.09 + 416.13 = 420.22 seconds
